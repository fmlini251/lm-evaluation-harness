"""Profile per-step GPU memory inside a single decoder layer for Llama 3.1 70B."""
import torch, gc, sys
sys.path.insert(0, "/home/howonlee/ozaki_npu")

device = torch.device("cuda:0")

def mb(x): return f"{x/1e6:.1f} MiB"
def gb(x): return f"{x/1e9:.3f} GiB"

def snap(tag):
    torch.cuda.synchronize(device)
    a = torch.cuda.memory_allocated(device)
    p = torch.cuda.max_memory_allocated(device)
    print(f"  [{tag:>40}] alloc={gb(a):>10}  peak={gb(p):>10}  delta_alloc={gb(a - snap.prev):>10}")
    snap.prev = a
snap.prev = 0

def main():
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    from lm_eval.models.ozaki_hf import OzakiHFLM
    wrapper = OzakiHFLM(
        pretrained="meta-llama/Llama-3.1-70B",
        dtype="bfloat16",
        attn_implementation="eager",
        rslt_type="ozaki",
        s_lst="2",
        weight_cache=True,
        offloading=True,
        out_feature_ts=512,
        max_inflight_offloads=1,
        device="cuda:0",
    )
    model = wrapper.model
    model.eval()

    # ---- Setup: put permanent modules + layer 0 on GPU ----
    base_model = model.model
    from emulation.llm.utils import _load_layer_selective
    base_model.embed_tokens.to(device)
    base_model.rotary_emb.to(device)
    model.lm_head.to(device)
    base_model.norm.to(device)
    _load_layer_selective(base_model.layers[0], device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect(); torch.cuda.empty_cache()
    snap.prev = torch.cuda.memory_allocated(device)

    print(f"\n=== Baseline (permanent + layer 0): {gb(snap.prev)}")
    print(f"=== Profiling single layer 0 forward, batch=8, seq=512 ===\n")

    # ---- Prepare inputs ----
    batch, seq = 8, 512
    input_ids = torch.randint(0, 128256, (batch, seq), device=device)
    inputs_embeds = base_model.embed_tokens(input_ids)
    position_ids = torch.arange(seq, device=device).unsqueeze(0)
    position_embeddings = base_model.rotary_emb(inputs_embeds, position_ids)
    cache_position = torch.arange(seq, device=device)

    from transformers.masking_utils import create_causal_mask
    causal_mask = create_causal_mask(
        config=base_model.config, input_embeds=inputs_embeds,
        attention_mask=None, cache_position=cache_position,
        past_key_values=None, position_ids=position_ids,
    )
    torch.cuda.synchronize(device)
    snap("inputs_ready")

    # ---- Manual layer 0 forward with memory tracking ----
    layer = base_model.layers[0]
    hidden = inputs_embeds

    # 1) input_layernorm
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    snap("input_layernorm")

    # 2) self_attn: manual decomposition
    attn = layer.self_attn
    bsz, q_len, _ = hidden.size()

    # Q/K/V projections
    q = attn.q_proj(hidden)
    snap("q_proj")
    k = attn.k_proj(hidden)
    snap("k_proj")
    v = attn.v_proj(hidden)
    snap("v_proj")

    # reshape to multi-head
    from transformers.models.llama.modeling_llama import repeat_kv
    q = q.view(bsz, q_len, attn.config.num_attention_heads, attn.head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, attn.config.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, attn.config.num_key_value_heads, attn.head_dim).transpose(1, 2)

    # Apply rotary
    import transformers.models.llama.modeling_llama as llama_mod
    cos, sin = position_embeddings
    q, k = llama_mod.apply_rotary_pos_emb(q, k, cos, sin)
    snap("rotary_applied")

    # repeat_kv
    k_expanded = repeat_kv(k, attn.num_key_value_groups)
    v_expanded = repeat_kv(v, attn.num_key_value_groups)
    snap("repeat_kv")
    print(f"    K expanded: {k_expanded.shape} = {gb(k_expanded.nelement()*k_expanded.element_size())}")
    print(f"    V expanded: {v_expanded.shape} = {gb(v_expanded.nelement()*v_expanded.element_size())}")

    # Q*K^T via ozaki batched_gemm
    import copy
    from emulation.llm.ozaki_matmul import batched_gemm
    custom_gemm_config = attn.q_proj.custom_gemm_config
    ozaki_config = attn.q_proj.ozaki_config

    scaling = attn.head_dim ** -0.5
    attn_cfg = copy.copy(custom_gemm_config)
    attn_cfg.name = custom_gemm_config.name + ".attn_weights"

    q_flat = q.reshape(-1, q_len, attn.head_dim)    # (512, 512, 128)
    k_flat = k_expanded.reshape(-1, q_len, attn.head_dim).transpose(1, 2)  # (512, 128, 512)
    snap("before_QKT")
    print(f"    q_flat: {q_flat.shape}, k_flat: {k_flat.shape}")
    print(f"    batched_gemm b={q_flat.shape[0]}, m={q_flat.shape[1]}, k={q_flat.shape[2]}, n={k_flat.shape[2]}")

    attn_weights = batched_gemm(q_flat, k_flat, custom_gemm_config=attn_cfg, ozaki_config=ozaki_config)
    snap("after_QKT (!!)")
    print(f"    attn_weights: {attn_weights.shape} = {gb(attn_weights.nelement()*attn_weights.element_size())}")

    attn_weights = attn_weights.to(q.dtype) * scaling
    snap("after_QKT_scale")

    # Mask + softmax
    if causal_mask is not None:
        cm = causal_mask[:, :, :, :q_len]
        attn_weights = attn_weights + cm.expand(bsz, attn.config.num_attention_heads, q_len, q_len).reshape(-1, q_len, q_len)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    snap("after_softmax")

    # attn * V via ozaki
    attn_cfg2 = copy.copy(custom_gemm_config)
    attn_cfg2.name = custom_gemm_config.name + ".attn_output"
    v_flat = v_expanded.reshape(-1, q_len, attn.head_dim)
    attn_out = batched_gemm(attn_weights, v_flat, custom_gemm_config=attn_cfg2, ozaki_config=ozaki_config)
    snap("after_attnV")

    attn_out = attn_out.to(q.dtype).view(bsz, attn.config.num_attention_heads, q_len, attn.head_dim)
    attn_out = attn_out.transpose(1, 2).contiguous()
    snap("attn_out_reshape")

    # o_proj
    attn_out = attn.o_proj(attn_out)
    snap("o_proj")

    # residual
    hidden = residual + attn_out
    del residual, attn_out, q, k, v, k_expanded, v_expanded, attn_weights, q_flat, k_flat, v_flat
    gc.collect(); torch.cuda.empty_cache()
    snap("post_attn_cleanup")

    # 3) MLP
    residual2 = hidden
    hidden = layer.post_attention_layernorm(hidden)
    snap("post_attn_layernorm")

    gate = layer.mlp.gate_proj(hidden)
    snap("gate_proj")
    up = layer.mlp.up_proj(hidden)
    snap("up_proj")
    act = layer.mlp.act_fn(gate) * up
    del gate, up
    snap("act_fn * up")
    down = layer.mlp.down_proj(act)
    del act
    snap("down_proj")
    hidden = residual2 + down
    del down, residual2
    gc.collect(); torch.cuda.empty_cache()
    snap("post_mlp_cleanup")

    print(f"\n=== Final peak: {gb(torch.cuda.max_memory_allocated(device))} ===")

if __name__ == "__main__":
    main()
