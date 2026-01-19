import os
import argparse

import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import sys
import gc
import torch.nn.functional as F
from scipy.stats import pearsonr
from typing import List, Dict, Optional

try:
    from ScafVAE.app.app_utils import ScafVAEBase, load_ModelBase
    from ScafVAE.utils.dataset_utils import ScafDataset, collate_ligand
    from models.DEGMON.DEGMON import DEGMON_12014_VAE
    from models.DEGMON.DEG_AE import GO_Autoencoder
    from models.flow.MLP import GatedConditionalFlowMLP
    from utils.evaluation import *
    from torchdiffeq import odeint
except ImportError as e:
    print(f"Import Error: {e}"); sys.exit(1)

class DEGDataset(Dataset):
    def __init__(self, deg_df, first_matrix_path, split_name='unknown'):
        self.split_name = split_name
        print(f"   Applying gene order to '{split_name}' split...")
        
        print(f"   🧬 Loading gene order from: {first_matrix_path}")
        first_matrix_connection = pd.read_csv(first_matrix_path, index_col=0)
        ordered_gene_names = list(first_matrix_connection.index)
        print(f"       ✅ Loaded {len(ordered_gene_names)} genes")
        
        try:
            deg_only_df = deg_df[ordered_gene_names].astype(np.float32)
        except KeyError as e:
            print(f"💥 ERROR: Some genes not found in the dataframe.")
            raise
            
        deg_data_map = pd.concat([deg_df['cmap_name'], deg_only_df], axis=1)
        deg_data_map = deg_data_map.set_index('cmap_name')
        
        self.deg_data = deg_data_map.values.astype(np.float32)
        self.sample_names = deg_data_map.index.tolist()

    def __getitem__(self, idx):
        deg_tensor = torch.from_numpy(self.deg_data[idx])
        sample_name = self.sample_names[idx]
        return deg_tensor, sample_name

    def __len__(self):
        return len(self.deg_data)


def collate_deg(batch):
    deg_list = [item[0] for item in batch]
    sample_names = [item[1] for item in batch]
    collated_deg_batch = torch.stack(deg_list, dim=0)
    return collated_deg_batch, sample_names


def create_dataloader(data_type, batch_size, num_workers, first_matrix_path):
    test_path = f'data/{args.data_type}/extra_test.feather'
    test_df = pd.read_feather(test_path)
    
    test_dataset = DEGDataset(test_df, first_matrix_path, split_name='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=False, collate_fn=collate_deg)
    
    print(f"   ✅ DataLoader ready: {len(test_dataset)} test samples")
    
    return test_loader, test_df


class ODEFunc(torch.nn.Module):
    def __init__(self, flow_model, condition_vector, guidance_scale=1.0, conditional=True):
        super().__init__()
        self.flow_model = flow_model
        self.condition_vector = condition_vector
        self.guidance_scale = guidance_scale
        self.conditional = conditional
        
    def forward(self, t, x):
        if isinstance(t, float):
            t = torch.tensor([t], device=x.device)
        
        t_batch = t.expand(x.size(0), 1)
        
        if self.conditional:
            v_cond = self.flow_model(x, t_batch, self.condition_vector)
            if hasattr(self.flow_model, 'null_condition'):
                null_cond = self.flow_model.null_condition.expand(x.size(0), -1)
                v_uncond = self.flow_model(x, t_batch, null_cond)
                v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond
        else:
            v = self.flow_model(x, t_batch)
            
        return v


@torch.no_grad()
def sample_molecules(args, flow_model, deg_model, scaf_vae, test_loader, test_df, device, output_dir):

    print(f"\n🎲 Generating molecules (Memory Optimized)...")
    print(f"   Mode: {'Conditional' if args.conditional else 'Unconditional'}")
    print(f"   Molecules per test sample: {args.num_samples}")
    print(f"   Solver: {args.solver}, Steps: {args.num_steps}")
    if args.conditional:
        print(f"   Guidance scale: {args.guidance_scale}")
    print(f"   Generation batch size: {args.generation_batch_size}")
    
    flow_model.eval()
    deg_model.eval()
    scaf_vae.eval()
    
    global_sample_idx = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {}
    total_generated = 0
    total_samples = len(test_df)
    
    with tqdm(total=total_samples, desc="Generating molecules", unit="sample") as pbar:
        for batch_idx, (deg_data, sample_names) in enumerate(test_loader):
            deg_data = deg_data.to(device, non_blocking=True)
            batch_size_actual = deg_data.size(0)
                        
            outputs = deg_model(deg_data)

            if isinstance(outputs, tuple):
                if len(outputs) == 4:
                    _, mu_deg, _, _ = outputs
                elif len(outputs) >= 2:
                    mu_deg = outputs[1]
                else:
                    mu_deg = outputs[0]
            else:
                mu_deg = outputs
            
            logvar_deg = None
                            
            for sample_idx in range(batch_size_actual):
                sample_name = sample_names[sample_idx]
                test_smiles = test_df.iloc[global_sample_idx]['cmap_name']
                sample_mols = []
                
                single_mu = mu_deg[sample_idx:sample_idx+1]
                if logvar_deg is not None:
                    single_logvar = logvar_deg[sample_idx:sample_idx+1]
                else:
                    single_logvar = None
                
                chunk_size = args.generation_batch_size
                num_chunks = (args.num_samples + chunk_size - 1) // chunk_size
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, args.num_samples)
                    current_chunk_size = chunk_end - chunk_start
                    
                    if args.conditional:
                        z = deg_model.reparameterize(single_mu, single_logvar)
                        if args.normalize_condition:
                            z = F.normalize(z, p=2, dim=1)
                        z_condition = z.repeat(current_chunk_size, 1)
                        x0 = torch.randn(current_chunk_size, args.latent_dim, device=device)
                    else:
                        z_samples = []
                        for _ in range(current_chunk_size):
                            z_sample = deg_model.reparameterize(single_mu, single_logvar)
                            z_samples.append(z_sample)
                        x0 = torch.cat(z_samples, dim=0)
                        z_condition = None
                    
                    # ODE Solver
                    if args.solver == 'dopri5':
                        ode_func = ODEFunc(flow_model, z_condition, args.guidance_scale, args.conditional)
                        integration_times = torch.tensor([0., 1.], device=device)
                        solution = odeint(ode_func, x0, integration_times, method='dopri5', atol=1e-5, rtol=1e-5)
                        final_latents = solution[1]
                    else:
                        dt = 1.0 / args.num_steps
                        x = x0.clone()
                        
                        for step in range(args.num_steps):
                            t = step * dt
                            t_tensor = torch.tensor([t], device=device)
                            ode_func = ODEFunc(flow_model, z_condition, args.guidance_scale, args.conditional)
                            
                            if args.solver == 'rk4':
                                k1 = ode_func(t_tensor, x)
                                k2 = ode_func(t_tensor + 0.5 * dt, x + 0.5 * dt * k1)
                                k3 = ode_func(t_tensor + 0.5 * dt, x + 0.5 * dt * k2)
                                k4 = ode_func(t_tensor + dt, x + dt * k3)
                                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                            elif args.solver == 'heun':
                                k1 = ode_func(t_tensor, x)
                                k2 = ode_func(t_tensor + dt, x + dt * k1)
                                x = x + (dt / 2.0) * (k1 + k2)
                            else:  # euler
                                v = ode_func(t_tensor, x)
                                x = x + dt * v
                        
                        final_latents = x
                    
                    mol_dict = scaf_vae.frag_decoder.sample(
                        batch_size=final_latents.size(0),
                        input_noise=final_latents,
                        output_smi=True
                    )
                    smiles_list = mol_dict.get('smi', [])
                    
                    for smi in smiles_list:
                        total_generated += 1
                        if smi is not None and smi != "None" and smi != "INVALID":
                            mol = Chem.MolFromSmiles(smi)
                            if mol is not None:
                                sample_mols.append(mol)
                            else:
                                sample_mols.append(None)
                        else:
                            sample_mols.append(None)
                    
                    del x0, final_latents, mol_dict, smiles_list
                    if z_condition is not None:
                        del z_condition
                    torch.cuda.empty_cache()
                
                key = f"{test_smiles}_{global_sample_idx}"
                results_dict[key] = {'generated_mols': sample_mols}
                
                global_sample_idx += 1
                
                pbar.update(1)
            
            del deg_data, mu_deg, logvar_deg
            if batch_idx % 10 == 0:
                gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n✅ Generation complete!")
    print(f"   Total test samples: {len(results_dict)}")
    
    pickle_file = os.path.join(output_dir, f'{args.data_type}_generated_molecules_dict_{args.guidance_scale}.pkl')
    print(f"\n💾 Saving results dictionary to {pickle_file}...")
    with open(pickle_file, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved {len(results_dict)} test samples with generated molecules")
    
    print("\n📋 Sample entries:")
    for i, (key, value) in enumerate(list(results_dict.items())[:3]):
        parts = key.rsplit('_', 1)
        smiles_part = parts[0][:50]
        sample_num = parts[1]
        print(f"   Sample {sample_num}: {smiles_part}... → {len(value['generated_mols'])} mols")
    
    return results_dict


def main(args):
    print("\n" + "="*70)
    print("🔬 Molecular Generation Model Evaluation (Memory Optimized)")
    print("="*70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    # 모델 로드
    print("\n🧠 Loading models...")
    if args.conditional:
        flow_model = GatedConditionalFlowMLP(
            embedding_dim=args.latent_dim, 
            condition_dim=args.latent_dim, 
            model_dim=args.model_dim, 
            num_layers=args.num_layers, 
            combine_method=args.combine_method,
            dropout=args.dropout
        ).to(device)
            
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Flow Model loaded")

    deg_model = GO_Autoencoder(dims=[12014, 1574, 1386, 951, 515], latent_dim=args.latent_dim).to(device)
    deg_model.load_state_dict(torch.load(args.deg_vae_path, map_location=device)['model_state_dict'])
    
    scaf_vae = ScafVAEBase().to(device)
    scaf_chk = load_ModelBase()
    scaf_vae.load_state_dict(scaf_chk['model_state_dict'])
    scaf_vae_args = scaf_chk['args']
    print("   ✅ DEG VAE and ScafVAE loaded")

    print("\n📂 Loading data...")
    test_loader, test_df = create_dataloader(args.data_type, args.batch_size, 4, args.gene_list_path)
    print(f"   Test samples (DEG): {len(test_loader.dataset)}")
    
    output_dir = os.path.dirname(args.model_checkpoint) if os.path.dirname(args.model_checkpoint) else '.'
    os.makedirs(output_dir, exist_ok=True)    

    results_dict = sample_molecules(
        args, flow_model, deg_model, scaf_vae, test_loader, test_df, device, output_dir
    )
    
    result_dict_file = os.path.join(output_dir, f'{args.data_type}_generated_molecules_dict_{args.guidance_scale}.pkl')
        
    del flow_model, deg_model, scaf_vae
    torch.cuda.empty_cache()
    gc.collect()
   
    print("\n" + "="*70)
    print("✅ Generation Complete!")
    print("="*70)
    print(f"Results saved to: {result_dict_file}")
    print(f"Total test samples: {len(results_dict)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow-based Model Evaluation (Memory Optimized)")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--deg_vae_path", type=str, default="checkpoints/DEGMON_AE_Best_model.pth")
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--combine_method', type=str, default='sum', choices=['concat', 'sum', 'cross_attn'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--task_path', type=str, default='../ScafVAE/ScafVAE/demo/CMAP_origianl/deg2mol_64dim')
    parser.add_argument('--gene_list_path', type=str, default="data/first_GO_matrix_cmap_12014x1574_deg랑유전자맞춤.csv")
    parser.add_argument("--num_samples", type=int, default=100, help="각 test 샘플당 생성할 분자 개수")
    parser.add_argument("--batch_size", type=int, default=256, help="DataLoader 배치 크기")
    parser.add_argument("--generation_batch_size", type=int, default=64, help="생성 시 청크 크기 (메모리 절약)")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--solver", type=str, default='euler', choices=['euler', 'heun', 'rk4', 'dopri5'])
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--normalize_condition", action='store_true')
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--eval_chunk_size", type=int, default=10000)
    parser.add_argument("--data_type", type=str, default='KO', choices=['KO', 'KD', 'Perturb-seq'])

    args = parser.parse_args()
    main(args)