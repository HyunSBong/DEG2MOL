import os
import argparse
import time
from typing import Any
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from scipy.optimize import linear_sum_assignment

# ==============================================================================
# 📂 1. SETUP & IMPORTS
# ==============================================================================
try:
    import wandb
except ImportError:
    wandb = None

import sys
try:
    from ScafVAE.app.app_utils import ScafVAEBase, load_ModelBase
    from ScafVAE.utils.dataset_utils import ScafDataset, collate_ligand
    from models.DEGMON.DEG_AE import GO_Autoencoder
    from utils.training_utils import AverageMeter, create_run_directory, save_config, save_checkpoint
    
    from models.flow.MLP import GatedConditionalFlowMLP
except ImportError as e:
    print(f"💥 모듈 임포트 실패: {e}")
    sys.exit(1)

def setup_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        print("pip install wandb")
        return None
    
    run_name = args.run_name
    
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        tags=args.wandb_tags,
        config=vars(args)
    )

# ==============================================================================
# 📂 1. EMA MANAGER CLASS
# ==============================================================================
class EMAManager:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def load_pretrained_models(args, device):
    scaf_vae = ScafVAEBase().to(device)
    chk = load_ModelBase()
    scaf_vae_args = chk['args']
    scaf_vae_args.is_main_process = True
    scaf_vae_args.rand_inp = False
    scaf_vae_args.n_batch = -1  # Use all data
    scaf_vae_args.persistent_workers = False
    scaf_vae.load_state_dict(chk['model_state_dict'])
    scaf_vae.eval()
    for param in scaf_vae.parameters():
        param.requires_grad = False

    if not os.path.exists(args.deg_vae_path):
        raise FileNotFoundError(f"DEGMON Error {args.deg_vae_path}")

    deg_model = GO_Autoencoder(dims=[12014, 1574, 1386, 951, 515], latent_dim=args.latent_dim).to(device)
    
    checkpoint = torch.load(args.deg_vae_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    deg_model.load_state_dict(state_dict)
    deg_model.eval()
    for param in deg_model.parameters():
        param.requires_grad = False
    
    return deg_model, scaf_vae, scaf_vae_args

def load_gene_list(path):
    """Load ordered gene list from a text file."""
    with open(path, 'r') as f:
        genes = [line.strip() for line in f]
    print(f"🧬 Loaded {len(genes)} ordered gene names.")
    return genes

# ==============================================================================
# 2. Dataset Class
# ==============================================================================
class DEGandScafDataset(ScafDataset):
    def __init__(self, deg_df, first_matrix_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        split_name = kwargs.get('split_name', 'unknown')
        print(f"⚙️ Initializing Dataset for '{split_name}' (One-to-Many Mode)...")
        
        first_matrix = pd.read_csv(first_matrix_path, index_col=0)
        ordered_gene_names = list(first_matrix.index)
        
        deg_values = deg_df[ordered_gene_names].values.astype(np.float32)
        self.deg_tensor_data = torch.from_numpy(deg_values)
        
        self.full_id_list = deg_df['cmap_name'].tolist()
        
        self.data_list = self.full_id_list
        self.sub_data_list = self.full_id_list
        
        if len(self.deg_tensor_data) != len(self.sub_data_list):
            raise RuntimeError(f"Length Mismatch! Tensor: {len(self.deg_tensor_data)} vs List: {len(self.sub_data_list)}")
            
        print(f"   ✅ Dataset Synced: {len(self.sub_data_list)} samples (with duplicates).")

    def __getitem__(self, idx):
        mol_data = super().__getitem__(idx)
        
        deg_tensor = self.deg_tensor_data[idx]
        
        loaded_id = mol_data['idx'] 
        expected_id = self.full_id_list[idx]
        
        if loaded_id != expected_id:
             raise RuntimeError(f"❌ DATA MISMATCH at idx {idx}: Loaded '{loaded_id}' vs Expected '{expected_id}'")
             
        return (mol_data, deg_tensor)

    def __len__(self):
        return len(self.deg_tensor_data)

# ==============================================================================
# 3. Data Loading Helpers
# ==============================================================================
def load_and_preprocess_data(args):
    print("📁 Loading and preprocessing data...")
    
    train_data = pd.read_feather(os.path.join(args.data_root, "train.feather"))
    val_data = pd.read_feather(os.path.join(args.data_root, "valid.feather"))

    scaf_dir = f'{args.task_path}/scaf'
    
    def get_valid_indices(df, latent_dir):
        valid_mask = []
        missing_count = 0
        for name in df['cmap_name']:
            file_path = os.path.join(latent_dir, f"{name}.npz")
            if os.path.exists(file_path):
                valid_mask.append(True)
            else:
                valid_mask.append(False)
                missing_count += 1
        return valid_mask, missing_count

    train_mask, train_miss = get_valid_indices(train_data, scaf_dir)
    train_filtered = train_data[train_mask].reset_index(drop=True)
    
    val_mask, val_miss = get_valid_indices(val_data, scaf_dir)
    valid_filtered = val_data[val_mask].reset_index(drop=True)

    print(f"   Filtering Report: Train missing {train_miss}, Valid missing {val_miss}")
    print("✅ Data filtering complete.")
    
    return train_filtered, valid_filtered

def collate_deg_and_ligand(batch):
    mol_list = [item[0] for item in batch]
    deg_list = [item[1] for item in batch]
    
    collated_mol_batch = collate_ligand(mol_list)
    collated_deg_batch = torch.stack(deg_list, dim=0)
    
    return collated_mol_batch, collated_deg_batch

def create_dataloaders(args, ScafVAE_args):
    train_filtered, valid_filtered = load_and_preprocess_data(args)
    
    def create_ds(df, split_name):
        dataset = DEGandScafDataset(
            df, 
            args.gene_list_path, 
            split_name, 
            ScafVAE_args,
            data_path=f'{args.task_path}/feat', 
            data_list=args.task_path, 
            scaf_path=f'{args.task_path}/scaf', 
            name='DEG2MOL'
        )
        print(f"   Dataset '{split_name}' created: {len(dataset)} samples.")
        return dataset

    train_dataset = create_ds(train_filtered, 'train')
    valid_dataset = create_ds(valid_filtered, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                              shuffle=True, collate_fn=collate_deg_and_ligand)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                            shuffle=False, collate_fn=collate_deg_and_ligand)
    
    print(f"✅ DataLoaders ready")
    
    return {
        'train_loader': train_loader, 'valid_loader': valid_loader,
        'train_dataset': train_dataset, 'val_dataset': valid_dataset,
        'train_df': train_filtered, 'valid_df': valid_filtered
    }

# ==============================================================================
# 🚀 4. TRAINING & VALIDATION LOOP
# ==============================================================================
def train_one_epoch(epoch, flow_model, deg_model, scaf_vae, train_loader, optimizer, device, args, scaler, ema=None):
    flow_model.train()
    meters = {'loss': AverageMeter()}
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", ncols=100)
    
    for mol_data, deg_tensor in pbar:
        deg_data = deg_tensor.to(device, non_blocking=True)
        mol_data_device = {k: v.to(device, non_blocking=True) for k, v in mol_data.items() if torch.is_tensor(v)}
        batch_size = deg_data.size(0)
        
        with autocast(enabled=args.use_amp):
            with torch.no_grad():
                out = deg_model(deg_data)
                c = out[1] # AE Latent
                                        
                vae_out = scaf_vae.frag_encoder(mol_data_device)
                x1 = vae_out.get('mu', vae_out.get('mean', vae_out.get('noise')))

            x0 = torch.randn_like(x1)
            dist = torch.cdist(x0.view(batch_size, -1), x1.view(batch_size, -1)) ** 2 
            row_ind, col_ind = linear_sum_assignment(dist.cpu().numpy())
            x0, x1, c = x0[row_ind], x1[col_ind], c[col_ind]

            t = torch.sigmoid(torch.randn(batch_size, 1, device=device)) * (1 - 2e-5) + 1e-5
            xt, ut = (1 - t) * x0 + t * x1, x1 - x0
            
            if torch.rand(1).item() < args.cfg_drop_prob:
                c = flow_model.null_condition.expand(batch_size, -1)
            
            v_pred = flow_model(xt, t, c)
            loss = F.mse_loss(v_pred, ut)

        optimizer.zero_grad()
        if args.use_amp:
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.gradient_clip > 0: torch.nn.utils.clip_grad_norm_(flow_model.parameters(), args.gradient_clip)
            optimizer.step()
        
        if ema is not None: ema.update()
            
        meters['loss'].update(loss.item(), batch_size)
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return {'loss': meters['loss'].avg}

@torch.no_grad()
def validate(epoch, flow_model, deg_model, scaf_vae, valid_loader, device, args, ema=None):
    if ema is not None: ema.apply_shadow()
    
    flow_model.eval()
    meters = {'loss': AverageMeter()}
    for mol_data, deg_tensor in valid_loader:
        deg_data = deg_tensor.to(device)
        mol_data_device = {k: v.to(device) for k, v in mol_data.items() if torch.is_tensor(v)}
        
        with autocast(enabled=args.use_amp):
            out = deg_model(deg_data)
            c = out[1]
                
            x1 = scaf_vae.frag_encoder(mol_data_device)['noise_mean']
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=device)
            xt, ut = (1 - t) * x0 + t * x1, x1 - x0
            v_pred = flow_model(xt, t, c)
            loss = F.mse_loss(v_pred, ut)
            
        meters['loss'].update(loss.item(), x1.size(0))
    
    if ema is not None: ema.restore()
    return {'loss': meters['loss'].avg}

# ==============================================================================
# 🚀 3. MAIN
# ==============================================================================
def main(args):
    print("\n" + "="*70 + "\n🚀 Conditional Flow Matching (Optimized Final)\n" + "="*70)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device} | Combine: {args.combine_method}")
    
    save_dir = create_run_directory(args.save_dir, args.run_name)
    save_config(args, save_dir)
    wandb_run = setup_wandb(args)
    
    deg_vae, scaf_vae, scaf_vae_args = load_pretrained_models(args, device)

    # 모델 선택
    flow_model = GatedConditionalFlowMLP(
        embedding_dim=args.latent_dim, 
        condition_dim=args.latent_dim, 
        model_dim=args.model_dim, 
        num_layers=args.num_layers, 
        combine_method=args.combine_method,
        dropout=args.dropout
    ).to(device)

    if not hasattr(flow_model, 'null_condition'):
        flow_model.register_buffer('null_condition', torch.zeros(1, args.latent_dim))

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema = EMAManager(flow_model, decay=args.ema_decay) if args.use_ema else None
    
    data_dict = create_dataloaders(
        args, scaf_vae_args
    )
    
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    scaler = GradScaler(enabled=args.use_amp)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n" + "="*70 + "\nTrain\n" + "="*70 + "\n")
    
    for epoch in range(args.num_epochs):
        train_metrics = train_one_epoch(epoch, flow_model, deg_vae, scaf_vae, data_dict['train_loader'], optimizer, device, args, scaler, ema)
        val_metrics = validate(epoch, flow_model, deg_vae, scaf_vae, data_dict['valid_loader'], device, args, ema)
        
        if scheduler: scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        is_best = val_metrics['loss'] < best_val_loss
        
        if is_best:
            best_val_loss, epochs_no_improve = val_metrics['loss'], 0
        else:
            epochs_no_improve += 1

        if wandb_run:
            wandb_run.log({'train/loss': train_metrics['loss'], 'val/loss': val_metrics['loss'], 'lr': curr_lr, 'epoch': epoch+1})

        if (epoch + 1) % args.save_interval == 0 or is_best:
            if ema is not None: ema.apply_shadow()
            save_checkpoint(epoch, flow_model, optimizer, scheduler, val_metrics, save_dir, is_best, max_keep=3)
            if ema is not None: ema.restore()

        print(f"Epoch {epoch+1:03d} | Train: {train_metrics['loss']:.5f} | Valid: {val_metrics['loss']:.5f} | LR: {curr_lr:.2e}{' 🎉' if is_best else ''}")
        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience: break

    if wandb_run: wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--task_path', type=str, default='../ScafVAE/ScafVAE/demo/CMAP_original/deg2mol_64dim')
    parser.add_argument('--gene_list_path', type=str, default="data/first_GO_matrix_cmap_12014x1574_deg랑유전자맞춤.csv")
    parser.add_argument('--deg_vae_path', type=str, default="checkpoints/DEGMON_AE_Best_model.pth")
    
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--combine_method', type=str, default='sum', choices=['sum', 'concat', 'cross_attn'])
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    parser.add_argument('--cfg_drop_prob', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--run_name', type=str, default=f'flow_final_{time.strftime("%Y%m%d-%H%M")}')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='deg2mol-flow')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=['Target-mu', 'Sum', 'GatedMLP'])
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    main(args)