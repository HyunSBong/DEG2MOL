import torch
import numpy as np
from typing import List, Set, Dict, Optional, Union
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Fraggle.FraggleSim import GetFraggleSimilarity
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from fcd_torch import FCD
from joblib import Parallel, delayed
import os

def smiles_to_mols(smiles_list: List[str]) -> List[Optional[Chem.Mol]]:
    """SMILES 리스트를 RDKit Mol 객체 리스트로 변환하고, 유효하지 않은 것은 None으로 둡니다."""
    return [Chem.MolFromSmiles(s) for s in smiles_list]

from rdkit import Chem
from typing import List, Set, Dict, Optional, Union
import itertools

def calculate_validity(mols: List[Optional[Chem.Mol]]) -> float:
    """전체 분자 중 유효한(Mol 객체인) 분자의 비율을 계산합니다."""
    if not mols:
        return 0.0
    valid_count = sum(1 for m in mols if m is not None)
    return valid_count / len(mols)

def calculate_uniqueness(valid_mols: List[Chem.Mol]) -> float:
    """유효한 분자 중 고유한 SMILES를 가진 분자의 비율을 계산합니다."""
    if not valid_mols:
        return 0.0
    # 유효한 Mol 객체만 SMILES로 변환하여 고유성 확인
    unique_smiles = set(Chem.MolToSmiles(m) for m in valid_mols if m is not None)
    return len(unique_smiles) / len(valid_mols)

def calculate_novelty(valid_mols: List[Chem.Mol], train_smiles_set: Set[str]) -> float:
    """유효한 분자 중 훈련 데이터셋에 없었던 새로운 분자의 비율을 계산합니다."""
    if not valid_mols:
        return 0.0
    
    # 훈련 데이터셋에 없는 새로운 분자만 카운트
    novel_count = 0
    total_valid = 0
    for m in valid_mols:
        if m is not None:
            smi = Chem.MolToSmiles(m)
            if smi is not None:
                total_valid += 1
                if smi not in train_smiles_set:
                    novel_count += 1
    
    return novel_count / max(1, total_valid)

def get_fingerprint(mol: Chem.Mol, fp_type: str) -> ExplicitBitVect:
    """분자 지문(fingerprint)을 계산합니다."""
    if fp_type == 'ecfp6':
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)  # ECFP6 (radius=3)
    elif fp_type == 'maccs':
        return MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")

def calculate_diversity(mols: List[Chem.Mol], fp_type: str) -> float:
    """내부 다양성 계산을 위한 Tanimoto 유사도 기반 지표를 계산합니다."""
    if len(mols) < 2:
        return 0.0
    
    fps = [get_fingerprint(m, fp_type) for m in mols]
    
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
            
    avg_sim = np.mean(similarities) if similarities else 1.0
    return 1.0 - avg_sim  # Diversity = 1 - Similarity

def calculate_similarity(gen_mols: List[Chem.Mol], ref_mols: List[Chem.Mol], fp_type: str) -> float:
    """생성된 분자와 참조(테스트) 데이터셋 간의 유사도를 계산합니다."""
    if not gen_mols or not ref_mols:
        return 0.0

    gen_fps = [get_fingerprint(m, fp_type) for m in gen_mols]
    ref_fps = [get_fingerprint(m, fp_type) for m in ref_mols]

    max_similarities = []
    for gen_fp in tqdm(gen_fps, desc=f"Calculating {fp_type} Similarity", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, ref_fps)
        max_similarities.append(max(sims))
        
    return np.mean(max_similarities)

# --- Fraggle Similarity (최적화됨) ---

def _calculate_max_fraggle_sim(gen_mol: Chem.Mol, ref_mols: List[Chem.Mol]) -> float:
    """Fraggle Similarity 병렬 처리를 위한 헬퍼 함수"""
    if gen_mol is None:
        return 0.0
        
    max_sim = 0.0
    for ref_mol in ref_mols:
        if ref_mol is None:
            continue
        try:
            sim, match = GetFraggleSimilarity(gen_mol, ref_mol)
            if sim > max_sim:
                max_sim = sim
        except Exception:
            continue
    return max_sim

def calculate_fraggle_similarity_optimized(
    gen_mols: List[Chem.Mol], 
    ref_mols: List[Chem.Mol], 
    n_jobs: int = -1
) -> float:
    """생성된 분자와 참조 데이터셋 간의 Fraggle 유사도를 병렬 처리로 계산합니다."""
    if not gen_mols or not ref_mols:
        return 0.0
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"Calculating Fraggle Similarity using {n_jobs} cores...")
    
    valid_ref_mols = [m for m in ref_mols if m is not None]
    if not valid_ref_mols:
        return 0.0

    max_similarities = Parallel(n_jobs=n_jobs, verbose=5, backend='multiprocessing')(
        delayed(_calculate_max_fraggle_sim)(gen_mol, valid_ref_mols) 
        for gen_mol in gen_mols
    )
    
    valid_similarities = [s for s in max_similarities if s > 0.0]
    return np.mean(valid_similarities) if valid_similarities else 0.0

def calculate_fraggle_similarity_per_reference(
    gen_mols_groups: List[List[Chem.Mol]], 
    ref_mols: List[Chem.Mol],
    n_jobs: int = -1
) -> float:
    """
    조건부 생성 평가: 각 참조 분자와 해당 생성 분자 그룹 간의 최대 유사도 계산
    
    Args:
        gen_mols_groups: [[G1_1...G1_100], [G2_1...G2_100], ...] (2D 리스트)
        ref_mols: [R1, R2, R3, ...] (1D 리스트)
    
    Returns:
        각 그룹별 max similarity의 평균
    """
    if not gen_mols_groups or not ref_mols:
        return 0.0
    
    if len(gen_mols_groups) != len(ref_mols):
        print(f"⚠️  Warning: 그룹 수({len(gen_mols_groups)})와 참조 분자 수({len(ref_mols)})가 다릅니다!")
        min_len = min(len(gen_mols_groups), len(ref_mols))
        gen_mols_groups = gen_mols_groups[:min_len]
        ref_mols = ref_mols[:min_len]
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"Calculating Fraggle Similarity (per reference) using {n_jobs} cores...")
    print(f"   Test samples: {len(ref_mols)}")
    print(f"   Generated per sample: {len(gen_mols_groups[0]) if gen_mols_groups else 0}")
    
    def calculate_max_sim_for_group(ref_mol: Chem.Mol, gen_group: List[Chem.Mol]) -> float:
        """하나의 참조 분자와 그 생성 분자 그룹 간의 최대 유사도"""
        if ref_mol is None:
            return 0.0
        
        max_sim = 0.0
        for gen_mol in gen_group:
            if gen_mol is None:
                continue
            try:
                sim, _ = GetFraggleSimilarity(ref_mol, gen_mol)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= 0.999:  # 조기 종료
                        break
            except Exception:
                continue
        
        return max_sim
    
    # 병렬 처리: 각 (참조, 생성그룹) 쌍에 대해
    max_similarities = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(calculate_max_sim_for_group)(ref_mol, gen_group)
        for ref_mol, gen_group in zip(ref_mols, gen_mols_groups)
    )
    
    valid_similarities = [s for s in max_similarities if s > 0.0]
    result = np.mean(valid_similarities) if valid_similarities else 0.0
    
    print(f"   Valid comparisons: {len(valid_similarities)}/{len(max_similarities)}")
    
    return result

def calculate_fraggle_similarity_per_reference_efficient(
    gen_groups_smiles: List[List[str]],  # ✅ SMILES로 전달 (Mol 아님!)
    ref_smiles: List[str],
    n_jobs: int = -1
) -> float:
    """
    메모리 효율적: SMILES를 전달하고 각 워커가 자체 변환
    """
    if not gen_groups_smiles or not ref_smiles:
        return 0.0
    
    if len(gen_groups_smiles) != len(ref_smiles):
        min_len = min(len(gen_groups_smiles), len(ref_smiles))
        gen_groups_smiles = gen_groups_smiles[:min_len]
        ref_smiles = ref_smiles[:min_len]
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    print(f"Fraggle Similarity (efficient parallel) using {n_jobs} cores...")
    print(f"   Test samples: {len(ref_smiles)}")
    
    # ✅ SMILES를 전달 (문자열은 가벼움)
    max_similarities = Parallel(
        n_jobs=n_jobs, 
        verbose=10, 
        backend='loky',
        batch_size=1  # 배치 크기 1로 메모리 최소화
    )(
        delayed(_calculate_max_sim_for_group_from_smiles)(ref_smile, gen_group_smiles)
        for ref_smile, gen_group_smiles in zip(ref_smiles, gen_groups_smiles)
    )
    
    valid_similarities = [s for s in max_similarities if s > 0.0]
    return np.mean(valid_similarities) if valid_similarities else 0.0


def _calculate_max_sim_for_group_from_smiles(
    ref_smile: str, 
    gen_group_smiles: List[str]
) -> float:
    """
    각 워커가 독립적으로 SMILES → Mol 변환
    (데이터 복사 없음, 필요할 때만 변환)
    """
    # 워커 내에서 변환
    ref_mol = Chem.MolFromSmiles(ref_smile)
    if ref_mol is None:
        return 0.0
    
    max_sim = 0.0
    for gen_smile in gen_group_smiles:
        if not gen_smile or gen_smile == "INVALID":
            continue
        
        gen_mol = Chem.MolFromSmiles(gen_smile)
        if gen_mol is None:
            continue
        
        try:
            sim, _ = GetFraggleSimilarity(ref_mol, gen_mol)
            if sim > max_sim:
                max_sim = sim
                if max_sim >= 0.999:
                    break
        except Exception:
            continue
    
    return max_sim

# --- 일반 FCD (Fréchet ChemNet Distance) ---

def calculate_fcd(
    gen_smiles: List[str], 
    ref_smiles: List[str],
    device: torch.device
) -> Optional[float]:
    """
    fcd-torch를 사용하여 일반 FCD 거리를 계산합니다.
    FCD(generated, reference) = 두 집합 간의 Fréchet ChemNet Distance
    """
    if not gen_smiles or not ref_smiles:
        return None
    
    try:
        fcd_calculator = FCD(device=device, n_jobs=8)
        fcd_distance = fcd_calculator(gen_smiles, ref_smiles)
        return float(fcd_distance)
    except Exception as e:
        print(f"FCD 계산 중 오류 발생: {e}")
        return None

# --- 전체 평가 함수 ---

def run_full_evaluation(
    generated_smiles: List[str], 
    train_smiles: List[str], 
    ref_smiles: List[str],
    device: torch.device
) -> Dict[str, float]:
    """
    생성된 SMILES에 대해 요청된 모든 지표를 계산하고 딕셔너리를 반환합니다.
    """
    print("Converting molecules to RDKit objects...")
    gen_mols = smiles_to_mols(generated_smiles)
    ref_mols = [m for m in smiles_to_mols(ref_smiles) if m is not None]
    
    # None (invalid molecule)은 제외하고 모두 지표 계산 (일관되게)
    valid_gen_mols = [m for m in gen_mols if m is not None]
    valid_gen_smiles = [Chem.MolToSmiles(m) for m in valid_gen_mols]

    metrics = {}
    # 전체에 대한 validity (즉, None 포함한 전체 중 valid 개수 비율)
    metrics['validity'] = len(valid_gen_mols) / len(gen_mols) if gen_mols else 0.0

    # # valid한 분자 없으면 나머지는 계산 불가/0 처리
    # if not valid_gen_mols:
    #     print("유효한 분자가 생성되지 않아 일부 지표를 계산할 수 없습니다.")
    #     metrics['fcd'] = float('nan')
    #     metrics['uniqueness'] = 0.0
    #     metrics['novelty'] = 0.0
    #     metrics['intdiv_ecfp6'] = 0.0
    #     metrics['similarity_ecfp6'] = 0.0
    #     metrics['similarity_maccs'] = 0.0
    #     metrics['similarity_fraggle'] = 0.0
    #     return metrics

    # FCD는 valid molecule SMILES만 사용
    # fcd_score = calculate_fcd(valid_gen_smiles, ref_smiles, device)
    # metrics['fcd'] = fcd_score if fcd_score is not None else float('nan')

    # 나머지 지표들도 valid molecule만
    metrics['uniqueness'] = calculate_uniqueness(valid_gen_mols)
    metrics['novelty'] = calculate_novelty(valid_gen_mols, set(train_smiles))
    # metrics['intdiv_ecfp6'] = calculate_diversity(valid_gen_mols, 'ecfp6')
    # metrics['similarity_ecfp6'] = calculate_similarity(valid_gen_mols, ref_mols, 'ecfp6')
    # metrics['similarity_maccs'] = calculate_similarity(valid_gen_mols, ref_mols, 'maccs')
    # metrics['similarity_fraggle'] = calculate_fraggle_similarity_optimized(
    #     valid_gen_mols, ref_mols, n_jobs=-1
    # )
    return metrics


def smiles_to_mols(smiles_list: List[str]) -> List[Optional[Chem.Mol]]:
    """SMILES 리스트를 RDKit Mol 객체 리스트로 변환합니다."""
    # (다른 함수에서 사용될 경우를 위해 유지)
    return [Chem.MolFromSmiles(s) for s in smiles_list]

def run_evaluation_basic(
    generated_data: List[Union[str, Chem.Mol, None]], # Mol 객체를 받을 수 있도록 변경
    train_smiles: List[str] 
) -> Dict[str, float]:
    """
    생성된 분자 데이터에 대해 validity, uniqueness, novelty만 계산합니다.
    """
    print("\n🔬 Calculating Basic Evaluation Metrics...")
    
    # 1. 입력 데이터 Mol 객체로 통일
    if generated_data and isinstance(generated_data[0], str):
        print("Input is SMILES. Converting to Mol objects...")
        # 이전에 0으로 만들었던 문제의 함수를 올바른 입력에 사용
        gen_mols = smiles_to_mols(generated_data)
    else:
        print("Input is RDKit Mol objects. Skipping conversion.")
        gen_mols = generated_data # 이미 Mol 객체 리스트 (또는 None 포함)

    # 2. 유효한 분자만 필터링
    valid_gen_mols = [m for m in gen_mols if m is not None]
    
    metrics = {}
    
    # 3. validity 계산
    metrics['validity'] = calculate_validity(gen_mols)

    if not valid_gen_mols:
        print("유효한 분자가 생성되지 않아 uniqueness와 novelty를 계산할 수 없습니다 (결과: 0.0)")
        metrics['uniqueness'] = 0.0
        metrics['novelty'] = 0.0
        return metrics

    # 4. uniqueness 계산
    metrics['uniqueness'] = calculate_uniqueness(valid_gen_mols)
    
    # 5. novelty 계산
    train_smiles_set = set(train_smiles) # 계산 효율을 위해 set으로 변환
    metrics['novelty'] = calculate_novelty(valid_gen_mols, train_smiles_set)
    
    return metrics

def evaluate_generated_mols(
    results_dict: Dict[str, Dict[str, List[Optional[Chem.Mol]]]],
    train_smiles: List[str], 
) -> Dict[str, float]:
    """
    sample_molecules의 결과 딕셔너리에서 데이터를 추출하여 기본 평가 지표를 계산합니다.
    """
    print("\n\n📊 Starting Final Evaluation...")
    
    # 1. 모든 생성 분자 리스트 추출
    # generated_mols는 Mol 객체 리스트이므로, 그대로 추출합니다.
    all_gen_mols_with_none = list(itertools.chain.from_iterable(
        [v['generated_mols'] for v in results_dict.values()]
    ))
    
    # 2. 기본 지표 계산 (Validity, Uniqueness, Novelty)
    basic_metrics = run_evaluation_basic(
        generated_data=all_gen_mols_with_none,
        train_smiles=train_smiles
    )
    
    return basic_metrics

