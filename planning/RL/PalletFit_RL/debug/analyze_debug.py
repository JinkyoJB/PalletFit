import torch
import torch.nn.functional as F
import sys
import os

# =========================================================
# 분석할 파일 경로를 여기에 입력하세요
# =========================================================
DEBUG_FILE_PATH = "planning/RL/PalletFit_RL/debug/ppo_mask_debug_1765419330_in_.pt"
# =========================================================

def analyze_debug_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"🔍 분석 시작: {file_path}")
    try:
        data = torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # 1. 예외 메시지 확인
    print("\n" + "="*50)
    print("1. 발생한 예외 메시지 (Exception)")
    print("="*50)
    print(data.get("exc", "No exception message saved."))

    # 2. Action Masks 분석
    print("\n" + "="*50)
    print("2. Action Masks 분석")
    print("="*50)
    masks = data.get("action_masks", None)
    if masks is None:
        print("⚠️ action_masks가 저장되지 않았습니다.")
    else:
        print(f"Shape: {masks.shape}")
        print(f"Data Type: {masks.dtype}")
        
        # 모든 행동이 마스킹(False)된 배치가 있는지 확인
        if masks.ndim > 1:
            # 보통 (Batch, N_Actions) 또는 (Batch, N_Actions, 1)
            flatten_mask = masks.view(masks.shape[0], -1)
            valid_counts = flatten_mask.sum(dim=1)
            invalid_rows = (valid_counts == 0).nonzero(as_tuple=True)[0]
            
            print(f"Min Valid Actions per row: {valid_counts.min().item()}")
            print(f"Max Valid Actions per row: {valid_counts.max().item()}")
            
            if len(invalid_rows) > 0:
                print(f"❌ 경고: 모든 행동이 금지된(False) 행이 {len(invalid_rows)}개 있습니다!")
                print(f"   Indices: {invalid_rows.tolist()}")
            else:
                print("✅ 모든 행에 최소 1개 이상의 유효한 행동(True)이 존재합니다.")
        else:
            print("Action masks 차원이 예상과 다릅니다 (1D?).")

    # 3. Logits 분석 (Simplex Error의 주 원인)
    print("\n" + "="*50)
    print("3. Logits (신경망 출력값) 분석")
    print("="*50)
    
    # 두 가지 키로 저장되었을 가능성을 모두 확인
    logits = data.get("logits", None)
    if logits is None:
        logits = data.get("last_logits", None)
    
    if logits is None:
        print("⚠️ Logits 데이터가 없습니다!")
        print("💡 [원인 추정] custom_value_policy.py의 _get_action_dist_from_latent 함수에서")
        print("   self._last_logits = logits 구문이 없어서 저장되지 않았을 수 있습니다.")
    else:
        print(f"Shape: {logits.shape}")
        
        # NaN / Inf 검사
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        if has_nan:
            print("❌ CRITICAL: Logits에 NaN이 포함되어 있습니다.")
        elif has_inf:
            print("❌ CRITICAL: Logits에 Inf(무한대)가 포함되어 있습니다.")
        else:
            print("✅ Logits에 NaN이나 Inf는 없습니다.")
            print(f"Max Value: {logits.max().item():.4f}")
            print(f"Min Value: {logits.min().item():.4f}")
            print(f"Mean Value: {logits.mean().item():.4f}")

        # 4. 확률 합(Simplex) 검증
        if masks is not None and masks.shape[0] == logits.shape[0]:
            print("\n" + "="*50)
            print("4. 마스킹 적용 후 확률 분포(Softmax) 검증")
            print("="*50)
            
            # 마스킹 적용 로직 재현 (sb3_contrib 방식)
            huge_neg = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            if masks.dim() == 3: # (B, N, 1) 대응
                 masks_flat = masks.squeeze(-1)
            else:
                 masks_flat = masks
                 
            # 마스크가 0(False)인 곳을 매우 작은 값으로 대체
            masked_logits = torch.where(masks_flat.bool(), logits, huge_neg)
            
            # Softmax 계산
            probs = F.softmax(masked_logits, dim=-1)
            sum_probs = probs.sum(dim=-1)
            
            print(f"Sum of probabilities (First 5): {sum_probs[:5].tolist()}")
            
            # 1.0과의 오차 확인
            diff = (sum_probs - 1.0).abs()
            max_diff = diff.max().item()
            
            print(f"Max deviation from 1.0: {max_diff:.8f}")
            
            if max_diff > 1e-3: # 오차가 너무 크면 문제
                print("❌ 확률의 합이 1.0에서 크게 벗어났습니다. (Numerical Instability or NaN propagation)")
                
                # 범인 찾기
                bad_indices = (diff > 1e-3).nonzero(as_tuple=True)[0]
                print(f"   문제가 되는 인덱스: {bad_indices.tolist()}")
                print(f"   해당 인덱스의 Logits: {logits[bad_indices[0]]}")
                print(f"   해당 인덱스의 Probs: {probs[bad_indices[0]]}")
            else:
                print("✅ 확률의 합이 정상 범위(approx 1.0) 내에 있습니다.")
        else:
            print("⚠️ Mask와 Logits의 배치가 맞지 않거나 데이터가 부족하여 확률 합 검증 불가.")

    # 5. Cand Embeddings 확인 (옵션)
    print("\n" + "="*50)
    print("5. Candidate Embeddings 확인")
    print("="*50)
    cand_emb = data.get("cand_emb", None)
    if cand_emb is not None:
        print(f"Shape: {cand_emb.shape}")
        if torch.isnan(cand_emb).any():
            print("❌ Candidate Embedding에 NaN이 있습니다. (입력 feature나 인코더 문제)")
        else:
            print("✅ Candidate Embedding 정상.")
    else:
        print("Candidate Embedding 정보 없음.")

if __name__ == "__main__":
    analyze_debug_file(DEBUG_FILE_PATH)