"""Verify unload_stream sync fix prevents OOM on Llama 3.1 8B."""
import torch, gc, sys
sys.path.insert(0, "/home/howonlee/ozaki_npu")

def gb(x): return f"{x/1e9:.3f} GiB"

def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from lm_eval.models.ozaki_hf import OzakiHFLM
    wrapper = OzakiHFLM(
        pretrained="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
        attn_implementation="eager",
        rslt_type="ozaki",
        s_lst="2",
        weight_cache=True,
        offloading=True,
        max_inflight_offloads=2,
        device="cuda:0",
    )
    model = wrapper.model
    model.eval()

    from emulation.llm.utils import prewarm_offloading_runtime
    prewarm_offloading_runtime(model)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    batch, seq = 8, 512
    input_ids = torch.randint(0, 128256, (batch, seq), device=device)

    print(f"=== Forward: batch={batch}, seq={seq}, 8B, s=2, weight_cache=True ===")

    for fwd_idx in range(3):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()
        before = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            output = model(input_ids)

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        alloc = torch.cuda.memory_allocated(device)
        print(f"\n--- Forward #{fwd_idx} ---")
        print(f"  Before:   {gb(before)}")
        print(f"  Peak:     {gb(peak)}")
        print(f"  After:    {gb(alloc)}")
        print(f"  Logits:   {output.logits.shape} = {gb(output.logits.nelement() * output.logits.element_size())}")
        del output

if __name__ == "__main__":
    main()
