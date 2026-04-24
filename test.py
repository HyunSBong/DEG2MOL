import os
import argparse
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import gc
import torch.nn.functional as F
from scipy.stats import pearsonr
from typing import List, Dict, Optional

try:
    from ScafVAE.app.app_utils import ScafVAEBase, load_ModelBase
    from ScafVAE.utils.dataset_utils import ScafDataset, collate_ligand
    from models.DEGMON.DEG_AE import GO_Autoencoder
    from models.flow.MLP import GatedConditionalFlowMLP
    from utils.evaluation import *
    from torchdiffeq import odeint
except ImportError as e:
    print(f"Import Error: {e}"); sys.exit(1)


def load_gene_list(path):
    """Load ordered gene list from a text file."""
    with open(path, 'r') as f:
        genes = [line.strip() for line in f]
    print(f"🧬 Loaded {len(genes)} ordered gene names.")
    return genes

def load_and_preprocess_data():
    print("📁 Loading and preprocessing data (MinMax Scaling disabled)...")
    
    # 데이터 로드
    train_data = pd.read_feather(os.path.join(args.data_root, "train.feather"))
    val_data = pd.read_feather(os.path.join(args.data_root, "test.feather"))

    scaf_dir = f'{args.task_path}/scaf'
    
    def find_missing_idx(idx_list, latent_dir=scaf_dir):
        missing_samples = [i for i in idx_list if not os.path.exists(os.path.join(latent_dir, f"{i}.npz"))]
        return missing_samples

    train_missing = find_missing_idx(train_data['cmap_name'])
    test_missing = find_missing_idx(val_data['cmap_name'])

    print(f"Missing scaffold files: train={len(train_missing)}, test={len(test_missing)}")

    train_filtered = train_data[~train_data['cmap_name'].isin(train_missing)].reset_index(drop=True)
    test_filtered = val_data[~val_data['cmap_name'].isin(test_missing)].reset_index(drop=True)
    
    print("✅ Data filtering and type conversion complete.")
    
    return train_filtered, test_filtered

class DEGandScafDataset(ScafDataset):
    def __init__(self, deg_df, first_matrix_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        print(f"Applying gene order standard to '{kwargs.get('split_name', 'unknown')}' split...")
        
        print(f"🧬 Loading gene order template from: {first_matrix_path}")
        first_matrix_connection = pd.read_csv(first_matrix_path, index_col=0)
        ordered_gene_names = list(first_matrix_connection.index)
        print(f"   ✅ Loaded {len(ordered_gene_names)} gene names.")
        
        try:
            deg_only_df = deg_df[ordered_gene_names].astype(np.float32)
            numeric_deg_df = deg_only_df.astype(np.float32)
        except KeyError as e:
            print(f"💥 ERROR: Some genes in the ordering standard were not found.")
            raise
            
        deg_data_map = pd.concat([deg_df['cmap_name'], numeric_deg_df], axis=1)
        deg_data_map = deg_data_map.set_index('cmap_name')

        self.deg_map = {
            name: torch.from_numpy(row.values)
            for name, row in deg_data_map.iterrows()
        }

    def __getitem__(self, idx):
        mol_data = super().__getitem__(idx)
        sample_name = mol_data['idx']
        deg_tensor = self.deg_map.get(sample_name, torch.zeros(len(next(iter(self.deg_map.values()))), dtype=torch.float32))
        return (mol_data, deg_tensor)

    def __len__(self):
        return len(self.sub_data_list)

def collate_deg_and_ligand(batch):
    mol_list = [item[0] for item in batch]
    deg_list = [item[1] for item in batch]
    collated_mol_batch = collate_ligand(mol_list)
    collated_deg_batch = torch.stack(deg_list, dim=0)
    return collated_mol_batch, collated_deg_batch

def create_dataloaders(ScafVAE_args, task_path, batch_size, num_workers, first_matrix_path):    
    train_filtered, test_filtered = load_and_preprocess_data()
    
    train_list = train_filtered['cmap_name'].tolist()
    test_list = test_filtered['cmap_name'].tolist()
    
    def create_dataset_with_proper_indexing(deg_df, split_name, data_list):
        dataset = DEGandScafDataset(
            deg_df, first_matrix_path, split_name, ScafVAE_args,
            data_path=f'{task_path}/feat', 
            data_list=task_path,
            scaf_path=f'{task_path}/scaf', 
            name='DEG2MOL'
        )
        
        dataset.data_list = data_list
        dataset.sub_data_list = data_list
        
        print(f"   {split_name} dataset: {len(dataset)} samples")
        return dataset

    train_dataset = create_dataset_with_proper_indexing(train_filtered, 'train', train_list)
    test_dataset = create_dataset_with_proper_indexing(test_filtered, 'test', test_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                              shuffle=True, collate_fn=collate_deg_and_ligand)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, collate_fn=collate_deg_and_ligand)
    
    print(f"   ✅ DataLoaders ready")
    return {
        'train_loader': train_loader, 'test_loader': test_loader,
        'train_dataset': train_dataset, 'val_dataset': test_dataset,
        'train_df': train_filtered, 'test_df': test_filtered
    }

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
    
    total_generated = 0
    test_generated = 0
    global_sample_idx = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_dict = {}
    
    total_samples = len(test_df)
    
    with tqdm(total=total_samples, desc="Generating molecules", unit="sample") as pbar:
        for batch_idx, (mol_data, deg_tensor) in enumerate(test_loader):
            deg_data = deg_tensor.to(device, non_blocking=True)
            batch_size_actual = deg_data.size(0)
            
            sample_names = mol_data['idx']
            
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
                test_smiles = test_df.iloc[global_sample_idx]['canonical_smiles']
                sample_mols = []
                
                single_mu = mu_deg[sample_idx:sample_idx+1]
                if logvar_deg is not None:
                    single_logvar = logvar_deg[sample_idx:sample_idx+1]
                else:
                    single_logvar = None
                
                chunk_size = args.generation_batch_size
                num_chunks = (args.num_samples + chunk_size - 1) // chunk_size
                
                sample_test = 0
                
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, args.num_samples)
                    current_chunk_size = chunk_end - chunk_start

                    if args.conditional:
                        # Conditional Generation
                        z = deg_model.reparameterize(single_mu, single_logvar)
                        
                        if args.normalize_condition:
                            z = F.normalize(z, p=2, dim=1)
                        
                        z_condition = z.repeat(current_chunk_size, 1)
                        x0 = torch.randn(current_chunk_size, args.latent_dim, device=device)
                    else:
                        # Unconditional Generation (Unused condition)
                        # AE의 경우 single_mu가 고정값이므로 다양성이 없을 수 있음
                        # Flow Matching에서 Unconditional은 보통 z_condition=None으로 처리
                        z_samples = []
                        for _ in range(current_chunk_size):
                            z_sample = deg_model.reparameterize(single_mu, single_logvar)
                            z_samples.append(z_sample)
                        
                        # Note: Unconditional인데 왜 condition latent를 x0로 쓰는지 확인 필요
                        # 보통은 x0 = torch.randn(...) 이고 z_condition = None 입니다.
                        # 기존 코드 로직 유지:
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
                    
                    # ScafVAE 디코딩
                    mol_dict = scaf_vae.frag_decoder.sample(
                        batch_size=final_latents.size(0),
                        input_noise=final_latents,
                        output_smi=True
                    )
                    smiles_list = mol_dict.get('smi', [])
                    
                    # SMILES 추출 및 Mol 객체로 변환
                    for smi in smiles_list:
                        total_generated += 1
                        if smi is not None and smi != "None" and smi != "INtest":
                            mol = Chem.MolFromSmiles(smi)
                            if mol is not None:
                                sample_mols.append(mol)
                                test_generated += 1
                                sample_test += 1
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
                pbar.set_postfix({
                    'test': f"{test_generated}/{total_generated}",
                    'rate': f"{test_generated/max(1,total_generated)*100:.1f}%",
                    'last': f"{sample_test}/{args.num_samples}"
                })
            
            del deg_data, mu_deg, logvar_deg
            if batch_idx % 10 == 0:
                gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\n✅ Generation complete!")
    print(f"   Total test samples: {len(results_dict)}")
    print(f"   Total generated: {total_generated}")
    print(f"   test SMILES: {test_generated}")
    print(f"   Intest: {total_generated - test_generated}")
    print(f"   Success rate: {test_generated/max(1, total_generated)*100:.2f}%")
    
    pickle_file = os.path.join(output_dir, 'generated_molecules_dict.pkl')
    print(f"\n💾 Saving results dictionary to {pickle_file}...")
    with open(pickle_file, 'wb') as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   Saved {len(results_dict)} test samples with generated molecules")
    
    print("\n📋 Sample entries:")
    for i, (key, value) in enumerate(list(results_dict.items())[:3]):
        test_count = sum(1 for m in value['generated_mols'] if m is not None)
        parts = key.rsplit('_', 1)
        smiles_part = parts[0][:50]
        sample_num = parts[1]
        print(f"   Sample {sample_num}: {smiles_part}... → {len(value['generated_mols'])} mols ({test_count} test)")
    
    return results_dict, total_generated, test_generated


def evaluate_in_chunks(generated_smiles, train_smiles, ref_smiles, device, chunk_size=10000):
    print(f"\n📊 Evaluating in chunks (chunk_size={chunk_size})...")
    
    if len(generated_smiles) > chunk_size:
        print(f"   Large dataset detected. Using sampling strategy...")
        import random
        sampled_gen = random.sample(generated_smiles, min(chunk_size, len(generated_smiles)))
        sampled_train = random.sample(train_smiles, min(chunk_size, len(train_smiles)))
        sampled_ref = random.sample(ref_smiles, min(chunk_size, len(ref_smiles)))
        
        print(f"   Sampled sizes - gen: {len(sampled_gen)}, train: {len(sampled_train)}, ref: {len(sampled_ref)}")
        metrics = run_full_evaluation(sampled_gen, sampled_train, sampled_ref, device)
    else:
        metrics = run_full_evaluation(generated_smiles, train_smiles, ref_smiles, device)
    
    return metrics


def load_smiles(file_path: str) -> List[str]:
    print(f"\n📖 Loading SMILES from {file_path}...")
    with open(file_path, 'rb') as f:
        smiles_list = [line.strip() for line in f if line.strip() and line.strip() != "INtest"]
    print(f"   Loaded {len(smiles_list)} test SMILES")
    return smiles_list


def load_generated_smiles(file_path, max_load=None):
    print(f"\n📖 Loading generated SMILES from {file_path}...")
    smiles_list = []
    with open(file_path, 'rb') as f:
        for i, line in enumerate(f):
            if max_load and i >= max_load:
                print(f"   Reached max_load limit of {max_load}. Loaded {len(smiles_list)} SMILES.")
                break
            line = line.strip()
            if line and line != "INtest":
                smiles_list.append(line)
    print(f"   Loaded {len(smiles_list)} test SMILES")
    return smiles_list

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
            dropout=args.dropout # 학습시 사용한 dropout 값 전달
        ).to(device)
            
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Flow Model loaded")

    deg_model = GO_Autoencoder(dims=[10280, 2011, 1614, 1075], latent_dim=args.latent_dim).to(device) # 7-5
    deg_model.load_state_dict(torch.load(args.deg_vae_path, map_location=device)['model_state_dict'])

    scaf_vae = ScafVAEBase().to(device)
    scaf_chk = load_ModelBase()
    scaf_vae.load_state_dict(scaf_chk['model_state_dict'])
    scaf_vae_args = scaf_chk['args']
    scaf_vae_args.is_main_process = True
    scaf_vae_args.rand_inp = False
    scaf_vae_args.n_batch = -1
    scaf_vae_args.persistent_workers = False
    scaf_vae.eval()
    print("   ✅ DEG VAE and ScafVAE loaded")

    print("\n📂 Loading data...")
    data_dict = create_dataloaders(scaf_vae_args, args.task_path, args.batch_size, 4, args.gene_list_path)
    test_loader = data_dict['test_loader']
    test_df = data_dict['test_df']
    
    train_smiles = data_dict['train_df']['canonical_smiles'].tolist()
    ref_smiles = test_df['canonical_smiles'].tolist()
        
    print(f"\n✅ Data loaded:")
    print(f"   Train SMILES: {len(train_smiles)}")
    print(f"   Test SMILES (reference): {len(ref_smiles)}")
    print(f"   Test samples (DEG): {len(test_loader.dataset)}")

    output_dir = os.path.dirname(args.model_checkpoint) if os.path.dirname(args.model_checkpoint) else '.'
    os.makedirs(output_dir, exist_ok=True)

    results_dict, total_generated, test_generated = sample_molecules(
        args, flow_model, deg_model, scaf_vae, test_loader, test_df, device, output_dir
    )
    
    result_dict_file = os.path.join(output_dir, 'generated_molecules_dict.pkl')

    del flow_model, deg_model, scaf_vae
    torch.cuda.empty_cache()
    gc.collect()

    print("\n📊 Calculating evaluation metrics...")
    metrics = evaluate_generated_mols(
        results_dict=results_dict,
        train_smiles=train_smiles
    )
    
    print("\n" + "="*70)
    print("✅ Generation Complete!")
    print("="*70)
    print(f"Results saved to: {result_dict_file}")
    print(f"Total test samples: {len(results_dict)}")   
    print(f"Total molecules generated: {total_generated}")
    print(f"test molecules: {test_generated}")
    print(f"Success rate: {test_generated/max(1,total_generated)*100:.2f}%")
    
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Molecular Generation Model Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {args.model_checkpoint}\n")
        f.write(f"Mode: {'Conditional' if args.conditional else 'Unconditional'}\n")
        f.write(f"Solver: {args.solver}\n")
        f.write(f"Num steps: {args.num_steps}\n")
        if args.conditional:
            f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Molecules per sample: {args.num_samples}\n")
        f.write(f"Generation batch size: {args.generation_batch_size}\n")
        f.write("Evaluation Metrics:\n")
        for key, value in metrics.items():
            f.write(f"   {key:<25}: {value:.4f}\n")
        f.write("="*70 + "\n")
        
    print(f"\n💾 Results saved: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEG2MOL Evaluation")
    parser.add_argument("--model_checkpoint", type=str, default = 'checkpoints/DEG2MOL_best_model.pt')
    parser.add_argument("--deg_vae_path", type=str, default = 'checkpoints/DEGMON_AE_best_model.pth')
    parser.add_argument("--conditional", action='store_true')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--combine_method', type=str, default='sum', choices=['concat', 'sum', 'cross_attn'])
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--task_path', type=str, default='../ScafVAE/ScafVAE/demo/CMAP_original/deg2mol_64dim')
    parser.add_argument('--gene_list_path', type=str, default="data/BP/gene_attribute_matrix_overlap_with_L1000.csv")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of molecules to generate per test sample")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument("--generation_batch_size", type=int, default=64, help="Chunk size for generation (to save memory)")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--solver", type=str, default='euler', choices=['euler', 'heun', 'rk4', 'dopri5'])
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--normalize_condition", action='store_true')
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to use for evaluation (None=all)")
    parser.add_argument("--eval_chunk_size", type=int, default=10000, help="Chunk size for evaluation")

    args = parser.parse_args()
    main(args)