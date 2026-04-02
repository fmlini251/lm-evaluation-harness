"""
Diagnostic: check if Ozaki model produces different logits
for the same tokens when padding is present (batch_size>1).
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    device = torch.device("cuda:0")
    model_name = "meta-llama/Llama-3.1-8B"

    print("=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("=== Loading model with Ozaki ===")
    # Use the same path as lm_eval ozaki-hf
    from lm_eval.models.ozaki_hf import OzakiHFLM
    lm = OzakiHFLM(
        pretrained=model_name,
        dtype="bfloat16",
        attn_implementation="eager",
        rslt_type="ozaki",
        s_lst="2",
        weight_cache=True,
        device=str(device),
    )
    model = lm.model
    model.eval()

    # Two sentences of different lengths
    text_short = "The cat sat on"
    text_long  = "The quick brown fox jumps over the lazy dog and then"

    enc_short = tokenizer.encode(text_short, return_tensors="pt")[0]
    enc_long  = tokenizer.encode(text_long, return_tensors="pt")[0]
    len_short = len(enc_short)
    len_long  = len(enc_long)
    print(f"short len={len_short}, long len={len_long}")

    # === Test 1: short sequence alone (batch_size=1, no padding) ===
    inp1 = enc_short.unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = model(inp1)
    logits_alone = out1.logits[0, :len_short].cpu().float()

    # === Test 2: short sequence batched with long (batch_size=2, right-padding) ===
    # Right-pad short to match long length (same as lm_eval's pad_and_concat)
    padded_short = torch.cat([enc_short, torch.zeros(len_long - len_short, dtype=torch.long)])
    batch_inp = torch.stack([padded_short, enc_long]).to(device)

    # Case A: WITHOUT attention_mask (what lm_eval does)
    with torch.no_grad():
        out2a = model(batch_inp)
    logits_batched_no_mask = out2a.logits[0, :len_short].cpu().float()

    # Case B: WITH proper attention_mask
    attn_mask = torch.ones(2, len_long, dtype=torch.long, device=device)
    attn_mask[0, len_short:] = 0  # mask padding for short sequence
    with torch.no_grad():
        out2b = model(batch_inp, attention_mask=attn_mask)
    logits_batched_with_mask = out2b.logits[0, :len_short].cpu().float()

    # === Compare ===
    diff_no_mask = (logits_alone - logits_batched_no_mask).abs()
    diff_with_mask = (logits_alone - logits_batched_with_mask).abs()

    print(f"\n=== Results for short sequence (first {len_short} positions) ===")
    print(f"Max diff (alone vs batched WITHOUT mask): {diff_no_mask.max().item():.6e}")
    print(f"Mean diff (alone vs batched WITHOUT mask): {diff_no_mask.mean().item():.6e}")
    print(f"Max diff (alone vs batched WITH mask):    {diff_with_mask.max().item():.6e}")
    print(f"Mean diff (alone vs batched WITH mask):   {diff_with_mask.mean().item():.6e}")

    # Check log-probabilities for a specific next token
    log_probs_alone = torch.nn.functional.log_softmax(logits_alone[-1], dim=-1)
    log_probs_no_mask = torch.nn.functional.log_softmax(logits_batched_no_mask[-1], dim=-1)
    log_probs_with_mask = torch.nn.functional.log_softmax(logits_batched_with_mask[-1], dim=-1)

    top5_alone = log_probs_alone.topk(5)
    print(f"\nTop-5 predictions at last real position (alone):")
    for idx, (val, tok) in enumerate(zip(top5_alone.values, top5_alone.indices)):
        print(f"  {tokenizer.decode([tok.item()])!r}: {val.item():.4f}")

    print(f"\nTop-5 predictions at last real position (batched, NO mask):")
    top5_no = log_probs_no_mask.topk(5)
    for idx, (val, tok) in enumerate(zip(top5_no.values, top5_no.indices)):
        print(f"  {tokenizer.decode([tok.item()])!r}: {val.item():.4f}")

    print(f"\nTop-5 predictions at last real position (batched, WITH mask):")
    top5_wm = log_probs_with_mask.topk(5)
    for idx, (val, tok) in enumerate(zip(top5_wm.values, top5_wm.indices)):
        print(f"  {tokenizer.decode([tok.item()])!r}: {val.item():.4f}")

    # Are the rankings the same?
    rank_match_no_mask = (top5_alone.indices == top5_no.indices).all().item()
    rank_match_with_mask = (top5_alone.indices == top5_wm.indices).all().item()
    print(f"\nTop-5 rank match (alone vs no_mask): {rank_match_no_mask}")
    print(f"Top-5 rank match (alone vs with_mask): {rank_match_with_mask}")

if __name__ == "__main__":
    main()
