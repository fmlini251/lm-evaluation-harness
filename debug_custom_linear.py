"""
Isolate: does a single CustomLinear produce different outputs
for the same row when it appears in batch=1 vs batch=2?
"""
import torch, sys, copy
sys.path.insert(0, "/home/howonlee/ozaki_npu/emulation")

from emulation.llm.ozaki_matmul import CustomGemmConfig, GlobalOzakiConfig, batched_gemm
from emulation.llm.utils import prepare_ozaki
from emulation.llm.ozaki_llama import CustomLinear

device = torch.device("cuda:0")

# Setup identical to lm_eval ozaki-hf with s=2, k=128
s_lst = [2]
k = 128
log2M, moduli, invM, NMi, M = prepare_ozaki(s_lst, k)
ozaki_config = GlobalOzakiConfig(
    s_lst=s_lst, log2M_tensor=log2M, moduli_tensor=moduli,
    NMi_tensor=NMi, M_tensor=M, invM_tensor=invM,
    rounding="round_half_away_from_0", scale_method="new_compressed",
    transfering_shift=False, selective_s_boost=False, target_patterns=[],
    shift_bits=7, M_frac_bits=8, weight_cache=True,
)
ozaki_config.to(device)

gemm_cfg = CustomGemmConfig(
    in_feature_ts=4096, out_feature_ts=4096, chunk_size=k,
    name="test_linear", track_mtx_acc=False, track_model_acc=False,
    get_statistics=False, rslt_type="ozaki",
)

# Create a CustomLinear (simulating a projection layer)
in_f, out_f = 4096, 4096
layer = CustomLinear(in_f, out_f, custom_gemm_config=copy.copy(gemm_cfg),
                     ozaki_config=ozaki_config, bias=False,
                     dtype=torch.bfloat16, device=device)
# Fill weight with something deterministic
with torch.no_grad():
    layer.weight.copy_(torch.randn(1, in_f, out_f, dtype=torch.bfloat16, device=device) * 0.01)

# Input: one "real" row
x_real = torch.randn(1, 5, in_f, dtype=torch.bfloat16, device=device)
# A "padding" row (zeros, like token_id=0 embedding after many layers)
x_pad = torch.randn(1, 5, in_f, dtype=torch.bfloat16, device=device) * 0.001

# === Test 1: batch=1 (just the real input) ===
layer._weight_cache = None  # clear cache
with torch.no_grad():
    y1 = layer(x_real.clone()).cpu().float()

# === Test 2: batch=2 (real + padding stacked) ===
layer._weight_cache = None  # clear cache
x_batch = torch.cat([x_real, x_pad], dim=0)  # (2, 5, 4096)
with torch.no_grad():
    y2_batch = layer(x_batch.clone()).cpu().float()
y2 = y2_batch[0:1]  # first element should match y1

diff = (y1 - y2).abs()
print(f"=== CustomLinear: batch=1 vs batch=2, same input row ===")
print(f"y1 shape: {y1.shape}, y2 shape: {y2.shape}")
print(f"Max diff:  {diff.max().item():.6e}")
print(f"Mean diff: {diff.mean().item():.6e}")
print(f"y1 [0,0,:5]: {y1[0,0,:5]}")
print(f"y2 [0,0,:5]: {y2[0,0,:5]}")

if diff.max().item() > 1e-3:
    print("\n>>> BUG CONFIRMED: CustomLinear gives different results for same input in different batch sizes")

    # Now test the raw batched_gemm directly
    print("\n=== Testing raw batched_gemm ===")
    from emulation.llm.ozaki_matmul import OzakiConfig
    s = 2
    _m_scalars = ozaki_config._M_scalars.get(s) if hasattr(ozaki_config, '_M_scalars') else None
    from emulation.llm.ozaki_matmul import extract_exponent_Mfrac

    W = layer.weight.data.clone()  # (1, in_f, out_f)

    # batch=1
    A1 = x_real.clone().contiguous()  # (1, 5, 4096)
    cfg1 = copy.copy(gemm_cfg)
    cfg1.name = "test_b1"
    r1 = batched_gemm(A1, W, custom_gemm_config=cfg1, ozaki_config=ozaki_config).cpu().float()

    # batch=2
    A2 = x_batch.clone().contiguous()  # (2, 5, 4096)
    cfg2 = copy.copy(gemm_cfg)
    cfg2.name = "test_b2"
    r2 = batched_gemm(A2, W.expand(2, -1, -1).contiguous(), custom_gemm_config=cfg2, ozaki_config=ozaki_config).cpu().float()

    diff2 = (r1 - r2[0:1]).abs()
    print(f"batched_gemm Max diff:  {diff2.max().item():.6e}")
    print(f"batched_gemm Mean diff: {diff2.mean().item():.6e}")
    print(f"r1[0,0,:5]: {r1[0,0,:5]}")
    print(f"r2[0,0,:5]: {r2[0,0,:5]}")

    # Test with weight_cache path (B=None)
    print("\n=== Testing with weight_cache path ===")
    cfg3 = copy.copy(gemm_cfg)
    cfg3.name = "test_wc_b1"
    r3, wc = batched_gemm(A1, W, custom_gemm_config=cfg3, ozaki_config=ozaki_config, return_weight_cache=True)
    r3 = r3.cpu().float()

    cfg4 = copy.copy(gemm_cfg)
    cfg4.name = "test_wc_b2"
    r4 = batched_gemm(A2, None, custom_gemm_config=cfg4, ozaki_config=ozaki_config, weight_cache=wc).cpu().float()

    diff3 = (r3 - r4[0:1]).abs()
    print(f"weight_cache Max diff:  {diff3.max().item():.6e}")
    print(f"weight_cache Mean diff: {diff3.mean().item():.6e}")
    print(f"r3[0,0,:5]: {r3[0,0,:5]}")
    print(f"r4[0,0,:5]: {r4[0,0,:5]}")
else:
    print("\n>>> CustomLinear is fine - issue is elsewhere (attention?)")
