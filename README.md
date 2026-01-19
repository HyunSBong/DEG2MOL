# DEG2MOL: Conditional Latent Flow Matching for Transcriptome-Guided De Novo Drug Design

## Abstract

![Model Architecture](figures/overview.png)

Motivation: Traditional de novo drug design prioritizes physicochemical properties, yet often overlooks the objective of modulating biological states. Consequently, phenotypic drug discovery (PDD) has emerged in de novo design, generating novel molecules conditioned on transcriptomic profiles for the desired biological activity. Although large-scale datasets have facilitated the application of deep learning models to PDD, current architectures encounter critical limitations, leading to limited molecular validity, structural redundancy, or high inference latency. Furthermore, reliance on restricted genes, cell lines, and standard evaluation protocols may limit the rigorous assessment of structural generalization and de novo design capabilities.

Results: We propose DEG2MOL, the first conditional latent flow matching framework for PDD that generates molecules by transforming Gaussian noise into molecular embeddings guided by Gene Ontology-informed differentially expressed genes (DEG) information. DEG2MOL achieved superior performance in generating valid and unique molecules with faster inference speed compared to baselines, maintaining a uniqueness score of 0.87 across both random and scaffold splits, confirming its capacity for de novo drug design rather than simple memorization. We substantiated the biological relevance of the generated molecules through molecular docking simulations, which confirmed robust binding interactions comparable to those of the reference drugs. DEG2MOL further demonstrated generalizability across knockdown and knockout profiles validated against known inhibitors, notably extending to single-cell Perturb-seq data. Overall, DEG2MOL establishes a robust framework for transcriptome-guided de novo drug design based solely on DEG profiles.

## Environment Setting

### Required Packages

```bash
# PyTorch (CUDA support recommended)
pip install torch torchvision torchaudio

# Flow Matching and ODE Solver
pip install torchdiffeq

# Data Processing
pip install pandas numpy scipy

# Molecular Processing and Evaluation
pip install rdkit

# Progress Display
pip install tqdm

# Optional: Experiment Tracking
pip install wandb
```

## Data

### Data Format

The project uses the following data formats:

1. **DEG Data** (`.feather` format)
   - Columns: `cmap_name` (molecule identifier), gene names (12,014 genes)
   - File locations: `data/{data_type}/train.feather`, `data/{data_type}/valid.feather`
   - Example data types: `KO`, `KD`, `Perturb-seq`

2. **Gene Order File** (`.csv` format)
   - File that defines the standard order of gene names
   - Default path: `data/first_GO_matrix_cmap_12014x1574.csv`
   - Gene names stored as index

3. **Molecular Latent Representations** (`.npz` format)
   - Molecular latent representations encoded by ScafVAE
   - File location: `{task_path}/scaf/{cmap_name}.npz`
   - One `.npz` file per molecule

4. **Molecular Feature Data** (`.npz` format)
   - Additional feature information for molecules
   - File location: `{task_path}/feat/{cmap_name}.npz`

### Data Directory Structure

```
data/
├── {data_type}/
│   ├── train.feather      # Training DEG data
│   └── valid.feather      # Validation DEG data
```

### Pre-trained Models

- **DEG Encoder**: Model that encodes DEG data into latent space
  - Default path: `checkpoints/DEGMON_AE_Best_model.pth`
  - Supports Autoencoder types

- **ScafVAE**: Molecular encoding/decoding model
  - Automatically loaded from ScafVAE library

## Implementation

### 1. Training

Train the Flow Matching model.

```bash
python train.py \
    --use_ema \
    --use_amp \
    --use_scheduler \
    --save_dir ./checkpoints
```

#### Key Parameters

- `--combine_method`: Condition combination method (`sum`, `concat`, `cross_attn`)
- `--use_ema`: Whether to use Exponential Moving Average
- `--use_amp`: Whether to use Mixed Precision Training
- `--cfg_drop_prob`: Classifier-free guidance dropout probability (default: 0.3)

### 2. Testing

Generate molecules and evaluate using the trained model.

```bash
python test.py \
    --num_samples 100 \
    --guidance_scale 3 \
    --conditional
```

#### Key Parameters

- `--conditional`: Enable conditional generation mode
- `--num_samples`: Number of molecules to generate per test sample
- `--guidance_scale`: Classifier-free guidance scale

### 3. Inference

Generate molecules for new DEG data using the trained model.

```bash
python inference.py \
    --model_checkpoint ./checkpoints/DEG2MOL_best_model.pth \
    --data_type Perturb-seq \
    --num_samples 100 \
    --guidance_scale 3 \
```

#### Key Parameters

- `--data_type`: Data type (`KO`, `KD`, `Perturb-seq`)

### Output Files

- **Training**: Checkpoint files are saved in `--save_dir`
- **Testing/Inference**: Generated molecule dictionary is saved as a `.pkl` file
  - Filename: `{data_type}_generated_molecules_dict_{guidance_scale}.pkl`
  - Format: `{sample_name}_{idx}: {'generated_mols': [list of Mol objects]}`

### Model Architecture

#### Gated Conditional Flow MLP

- **Input**: Molecular latent representation `x`, time `t`, DEG condition `c`
- **Structure**: 
  - Time embedding (Sinusoidal)
  - Condition combination (sum/concat/cross-attention)
  - Gated MLP blocks
  - Output projection
- **Features**: Residual connections, Layer normalization, Dropout support

#### DEG Encoder

- **AE Mode**: `GO_Autoencoder` - Autoencoder-based encoder
  - Architecture: `[12014, 1574, 1386, 951, 515] → latent_dim`
