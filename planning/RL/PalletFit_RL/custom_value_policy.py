# planning/RL/PalletFit_RL/custom_value_policy.py
from typing import Dict, Any, Tuple, Optional
import time
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from pathlib import Path

# -----------------------------
# Set encoders (교체 가능한 인터페이스)
# -----------------------------
class DeepSetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, pooling: str = "mean"):
        super().__init__()
        assert pooling in {"mean", "sum", "max"}
        self.pooling = pooling
        self.phi = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(hidden, out_dim), nn.ReLU())

    def forward(self, X: th.Tensor, M: th.Tensor | None = None) -> th.Tensor:
        H = self.phi(X)  # (B,K,H)
        if M is not None:
            H = H * M.unsqueeze(-1)
            if self.pooling == "sum":
                pooled = H.sum(dim=1)
            elif self.pooling == "mean":
                denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
                pooled = H.sum(dim=1) / denom
            else:  # max
                H = H + (M.unsqueeze(-1) - 1) * 1e9
                pooled, _ = H.max(dim=1)
        else:
            if self.pooling == "sum": pooled = H.sum(dim=1)
            elif self.pooling == "mean": pooled = H.mean(dim=1)
            else: pooled, _ = H.max(dim=1)
        return self.rho(pooled)  # (B, out_dim)

class MAB(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.n_heads = n_heads; self.d_head = d // n_heads

    def forward(self, X, Y, mask=None):  # Q=X, K=Y, V=Y
        B, Nx, d = X.shape; Ny = Y.size(1)
        q = self.q(X).view(B, Nx, self.n_heads, self.d_head).transpose(1, 2)      # (B,H,Nx,dh)
        k = self.k(Y).view(B, Ny, self.n_heads, self.d_head).transpose(1, 2)      # (B,H,Ny,dh)
        v = self.v(Y).view(B, Ny, self.n_heads, self.d_head).transpose(1, 2)      # (B,H,Ny,dh)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)                     # (B,H,Nx,Ny)
        if mask is not None:                                                       # Y 마스크
            att = att.masked_fill(mask[:,None,None,:]==0, -1e9)
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, Nx, d)               # (B,Nx,d)
        X = self.norm1(X + self.proj(out))
        X = self.norm2(X + self.ff(X))
        return X

class SAB(nn.Module):
    def __init__(self, d, n_heads=4, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([MAB(d, n_heads) for _ in range(n_blocks)])
    def forward(self, X, mask=None):
        for mab in self.blocks:
            X = mab(X, X, mask)
        return X

class ISAB(nn.Module):
    def __init__(self, d, m=16, n_heads=4, n_blocks=2):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m, d))  # 유도 포인트
        self.blocks = nn.ModuleList([
            nn.ModuleList([MAB(d, n_heads), MAB(d, n_heads)]) 
            for _ in range(n_blocks)
        ])
    def forward(self, X, mask=None):
        I = self.inducing.expand(X.size(0), -1, -1)  # (B,m,d)
        for mab_I, mab_X in self.blocks:
            I = mab_I(I, X, mask)                    # I attends to X
            X = mab_X(X, I, None)                    # X attends to I
        return X

class PMA(nn.Module):
    def __init__(self, d, k=1, n_heads=4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d))  # k개 시드
        self.mab = MAB(d, n_heads)
    def forward(self, X, mask=None):
        S = self.seed.expand(X.size(0), -1, -1)   # (B,k,d)
        return self.mab(S, X, mask)               # (B,k,d)

class SetTransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128,
                 d_model: int = 128, n_heads: int = 4,
                 n_blocks: int = 2, use_isab: bool = False, m: int = 16,
                 use_pma: bool = True, k_seeds: int = 1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.enc = (ISAB(d_model, m=m, n_heads=n_heads, n_blocks=n_blocks)
                    if use_isab else SAB(d_model, n_heads=n_heads, n_blocks=n_blocks))
        self.pma = PMA(d_model, k=k_seeds, n_heads=n_heads) if use_pma else None
        self.out = nn.Sequential(nn.Linear(d_model * k_seeds, out_dim), nn.ReLU())

    def forward(self, X: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        H = F.relu(self.in_proj(X))                 # (B,N,d)
        H = self.enc(H, M)                          # (B,N,d)
        if self.pma is not None:
            P = self.pma(H, M)                      # (B,k,d)
            P = P.reshape(P.size(0), -1)            # (B,k*d)
            return self.out(P)                      # (B,out_dim)
        
        if M is not None:
            denom = M.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (H * M.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = H.mean(dim=1)
        return self.out(pooled)


# -----------------------------
# CustomCombinedExtractor
# -----------------------------
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Dict obs:
      1) items_topk:  (K, ITEM_FEAT_DIM)
      2) items_mask:  (K,)
      3) globals:     (GLOBAL_FEAT_DIM,)
      4) preview_queue: (PREVIEW_MAX, 4)
      5) act_mask:    (ACTION_MAX_CANDIDATES,)
      6) act_cands:   (ACTION_MAX_CANDIDATES, PIVOT_FEAT_DIM)

    블록:
      [1+2] items_topk → (DeepSets | SetTransformer) → z_items (128)
      [3]   globals    → MLP → z_globals (128)
      [4]   preview_queue → 1D-Conv + GAP → z_queue (64)
      [5+6] act_cands  → per-candidate 임베딩 
    결합:
      return z                                  
    """
    def __init__(self, observation_space: spaces.Dict, item_encoder: str = "deepset"):
            super().__init__(observation_space, features_dim=1)

            # --- 각 부분 차원 (동적 획득) ---
            d_item   = observation_space["items_topk"].shape[-1]
            d_global = observation_space["globals"].shape[-1]
            d_queueC = observation_space["preview_queue"].shape[-1]   # 4
            d_cand   = observation_space["act_cands"].shape[-1]

            # --- [1+2] items encoder ---
            if item_encoder.lower() == "set-transformer":
                self.items_enc = SetTransformerEncoder(in_dim=d_item, out_dim=128)
            else:
                self.items_enc = DeepSetEncoder(in_dim=d_item, hidden=128, out_dim=128, pooling="mean")

            # --- [3] globals encoder ---
            self.glob_enc = nn.Sequential(
                nn.Linear(d_global, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU()
            )

            # --- [4] queue encoder (Conv1d: Local Pattern) ---
            self.queue_conv = nn.Conv1d(in_channels=d_queueC, out_channels=64, kernel_size=3, padding=1)
            self.queue_proj = nn.Linear(64, 64)

            # --- [5+6] candidates ---
            # (A) per-candidate encoder: (B, N, d_cand) -> (B, N, 128)
            self.cand_enc = nn.Sequential(
                nn.Linear(d_cand, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU()
            )
            self._last_cand_emb = None  # (B, N, 128) 캐시

            # (B) pooled summary -> z_cands
            self._z_items_dim   = 128
            self._z_globals_dim = 128
            self._z_queue_dim   = 64
            self._z_cands_dim   = 64   
            self.cand_pool = nn.Sequential(  
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, self._z_cands_dim), nn.ReLU()
            )

            # --- 최종 features_dim ---
            self._features_dim = self._z_items_dim + self._z_globals_dim + self._z_queue_dim + self._z_cands_dim  # 384

    def _encode_globals(self, g: th.Tensor) -> th.Tensor:
        return self.glob_enc(g)  # (B, 128)

    def _encode_queue(self, q: th.Tensor) -> th.Tensor:
        # q: (B, L, C) -> transpose -> (B, C, L)
        x = q.transpose(1, 2)           
        x = F.relu(self.queue_conv(x))  # (B, 64, L)
        x = x.mean(dim=-1)              # GAP -> (B, 64)
        return F.relu(self.queue_proj(x))

    def _encode_candidates(self, cands: th.Tensor) -> th.Tensor:
        B, N, D = cands.shape
        c = self.cand_enc(cands.view(B * N, D))  # (B*N, 128)
        return c.view(B, N, 128)                 # (B, N, 128)

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        z_items   = self.items_enc(obs["items_topk"], obs.get("items_mask", None))  # (B, 128)
        z_globals = self._encode_globals(obs["globals"])                            # (B, 128)
        z_queue   = self._encode_queue(obs["preview_queue"])                        # (B, 64)

        # debug용
        self.obs_act_mask = obs.get("act_mask", None)

        # 후보 처리: 캐시 + 마스크드 요약
        if "act_cands" in obs:
            cand_emb = self._encode_candidates(obs["act_cands"])  # (B, N, 128)
            self._last_cand_emb = cand_emb

            m = obs.get("act_mask", None)
            if m is not None:
                m = m.to(cand_emb.dtype)
                if m.dim() == 2:               # (B, N) -> (B, N, 1)
                    m = m.unsqueeze(-1)
                
                masked = cand_emb * m          # (B, N, 128)

                denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)   # (B, 1, 1)
                denom = denom.squeeze(-1)                            # (B, 1)

                pooled = masked.sum(dim=1) / denom                   # (B, 128)
            else:
                pooled = cand_emb.mean(dim=1)                        # (B, 128)

            z_cands = self.cand_pool(pooled)                         # (B, 64)
        else:
            self._last_cand_emb = None
            z_cands = th.zeros((z_items.size(0), self._z_cands_dim), device=z_items.device)

        # concat → (B, 384) == self._features_dim
        return th.cat([z_items, z_globals, z_queue, z_cands], dim=-1)
    

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(th.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if (not self.training) or self.p <= 0:
            return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

class GEGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, p_drop: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_hidden)
        self.proj = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1) 
        x = self.proj(F.gelu(b) * a)       
        return self.drop(x)

def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)  
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)

def build_rope_cache(T, Dh, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, Dh, 2, device=device).float() / Dh))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq) 
    return torch.cos(freqs), torch.sin(freqs)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p_drop_attn: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.p_drop_attn = p_drop_attn

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope_cache=None):
        B, T, D = x.shape
        qkv = self.qkv(x)                                 
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)

        q, k, v = map(reshape, (q, k, v))

        if rope_cache is not None:
            cos, sin = rope_cache  
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.p_drop_attn if self.training else 0.0
        ) 

        y = y.transpose(1, 2).contiguous().view(B, T, D) 
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 mlp_ratio: float = 4.0, p_drop: float = 0.0, drop_path: float = 0.0, norm='rms'):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if norm == 'rms' else nn.LayerNorm(d_model)
        self.attn  = MultiheadSelfAttention(d_model, n_heads, p_drop_attn=p_drop)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = RMSNorm(d_model) if norm == 'rms' else nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ff = GEGLU(d_model, hidden, d_model, p_drop)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rope_cache=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), attn_mask, rope_cache))
        x = x + self.drop_path2(self.ff(self.norm2(x)))
        return x


class FTTransformerBackbone(nn.Module):
    def __init__(self, feature_dim: int, n_tokens: int = 8,
                 d_model: int = 256, n_heads: int = 8, depth: int = 6,
                 mlp_ratio: float = 6.0, p_drop: float = 0.1, drop_path: float = 0.1,
                 out_pi: int = 256, out_vf: int = 256, use_rope: bool = True):
        super().__init__()
        self.n_tokens, self.d_model = n_tokens, d_model
        self.use_rope = use_rope

        self.token_proj = nn.Linear(feature_dim, n_tokens * d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb    = nn.Parameter(torch.zeros(1, 1 + n_tokens, d_model)) if not use_rope else None

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, p_drop, drop_path, norm='rms')
            for _ in range(depth)
        ])
        self.norm = RMSNorm(d_model)

        self.head_pi = nn.Sequential(nn.Linear(d_model, out_pi), nn.SiLU())
        self.head_vf = nn.Sequential(nn.Linear(d_model, out_vf), nn.SiLU())

        self.latent_dim_pi = out_pi
        self.latent_dim_vf = out_vf

        if self.pos_emb is not None:
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, z: torch.Tensor):
        B, F = z.shape
        x = self.token_proj(z).view(B, self.n_tokens, self.d_model)  # (B,T,D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B,1+T,D)

        rope_cache = None
        if self.use_rope:
            T = x.size(1)
            Dh = self.d_model // self.blocks[0].attn.n_heads
            Dh = Dh if Dh % 2 == 0 else Dh + 1 
            rope_cache = build_rope_cache(T, Dh, x.device)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        for blk in self.blocks:
            x = blk(x, attn_mask=None, rope_cache=rope_cache)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head_pi(cls_out), self.head_vf(cls_out)
    
    def forward_actor(self, z: torch.Tensor) -> torch.Tensor:
        h_pi, _ = self.forward(z)
        return h_pi

    def forward_critic(self, z: torch.Tensor) -> torch.Tensor:
        _, h_vf = self.forward(z)
        return h_vf

# =========== 정책: _build_mlp_extractor 오버라이드 ===========
class PointerPolicyFT(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *args, cand_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        # (Hpi + Hc) -> 1 점수
        self.scorer = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi + cand_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _build_mlp_extractor(self) -> None:
        feature_dim = int(self.features_dim)

        if isinstance(self.net_arch, dict):
            cfg = self.net_arch
        elif isinstance(self.net_arch, list) and self.net_arch and isinstance(self.net_arch[0], dict):
            cfg = self.net_arch[0]
        else:
            cfg = {}

        self.mlp_extractor = FTTransformerBackbone(
            feature_dim=feature_dim,
            n_tokens=int(cfg.get("ft_n_tokens", 8)),
            d_model=int(cfg.get("ft_d_model", 128)),
            n_heads=int(cfg.get("ft_n_heads", 4)),
            depth=int(cfg.get("ft_depth", 2)),
            mlp_ratio=float(cfg.get("ft_mlp_ratio", 4.0)),
            p_drop=float(cfg.get("ft_p_drop", 0.0)),
            out_pi=int(cfg.get("ft_out_pi", 128)),
            out_vf=int(cfg.get("ft_out_vf", 128)),
        )

    def evaluate_actions(
        self,
        obs: th.Tensor | dict[str, th.Tensor],
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        # 1) feature 추출
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # 2) 분포 생성
        distribution = self._get_action_dist_from_latent(latent_pi)

        # 3) 마스킹 적용
        if action_masks is not None:
            try:
                if isinstance(action_masks, torch.Tensor):
                    bad = ~(action_masks.bool().any(dim=-1))
                    if bad.any():
                        am = action_masks.clone()
                        am[bad, 0] = True
                        action_masks = am
                distribution.apply_masking(action_masks)

            except Exception as e:
                import torch as _debug_th 
                import time
                from pathlib import Path

                path = Path("planning/RL/PalletFit_RL/debug")
                path.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                
                debug_data = {
                    "exc": str(e),
                    "action_masks": action_masks.detach().cpu() if action_masks is not None else None,
                    "last_logits": getattr(self, "_last_logits", None), 
                    "cand_emb": getattr(self.features_extractor, "_last_cand_emb", None),
                }

                save_path = f"planning/RL/PalletFit_RL/debug/ppo_mask_debug_{ts}_in_evaluate_actions.pt"
                _debug_th.save(debug_data, save_path)
                print(f"🔥 Critical Error Saved to: {save_path}")
                raise e

        # 4) 로그확률/밸류/엔트로피
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy


    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        cand_emb = getattr(self.features_extractor, "_last_cand_emb", None)
        
        # 1. Logits 계산
        if cand_emb is None:
            logits = latent_pi.new_zeros((latent_pi.size(0), self.action_space.n))
        else:
            B, N, _ = cand_emb.shape
            z = latent_pi.unsqueeze(1).expand(B, N, latent_pi.size(-1))
            logits = self.scorer(torch.cat([z, cand_emb], dim=-1)).squeeze(-1)

        # ─────────────────────────────────────────────────────────────
        # [수정] 수치 안정화 (Simplex Error 방지 & Robustness)
        # ─────────────────────────────────────────────────────────────
        
        # 1) NaN/Inf 제거
        logits = torch.nan_to_num(logits, neginf=-1e5, posinf=1e5, nan=0.0)

        # 2) Log-Sum-Exp Trick (Overflow 방지)
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max

        # 3) 값 범위 제한 (Clipping)
        logits = torch.clamp(logits, min=-50.0, max=50.0)

        # 4) 디버깅용 저장
        self._last_logits = logits
        # ─────────────────────────────────────────────────────────────

        dist = self.action_dist  # MaskableCategorical
        dist.proba_distribution(action_logits=logits)
        return dist