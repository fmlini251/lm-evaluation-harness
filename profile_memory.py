"""Profile GPU peak memory for ozaki-hf offloading forward pass."""
import torch
import gc
import sys
sys.path.insert(0, "/home/howonlee/ozaki_npu")

def profile_peak_memory():
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps to 0
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    print(f"=== Initial: {torch.cuda.memory_allocated(device)/1e9:.3f} GiB allocated")

    # --- Load model (same as lm_eval would) ---
    from lm_eval.models.ozaki_hf import OzakiHFLM
    model_wrapper = OzakiHFLM(
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
    model = model_wrapper.model
    model.eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    baseline = torch.cuda.memory_allocated(device)
    print(f"\n=== After model load (before forward): {baseline/1e9:.3f} GiB allocated")
    print(f"    Peak so far: {torch.cuda.max_memory_allocated(device)/1e9:.3f} GiB")

    # --- Monkey-patch offloading_forward to add memory checkpoints ---
    from emulation.llm.utils import offloading_forward as orig_forward
    from types import MethodType

    original_forward = model.forward.__func__ if hasattr(model.forward, '__func__') else model.forward

    mem_log = []

    def log_mem(tag):
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        mem_log.append((tag, alloc, reserved, peak))
        print(f"  [{tag:>30}] alloc={alloc/1e9:.3f} GiB  reserved={reserved/1e9:.3f} GiB  peak={peak/1e9:.3f} GiB")

    def profiled_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                         past_key_values=None, inputs_embeds=None, labels=None,
                         use_cache=None, cache_position=None, logits_to_keep=0, **kwargs):
        from emulation.llm.utils import (
            _load_layer_selective, _move_weight_cache_for_layer,
            _record_stream_for_module, _record_stream_for_weight_cache,
            _release_module_cached_weights, _clear_dequant_cache,
            _unload_layer_to_pinned, _unload_weight_cache_to_pinned,
            _pin_all_cpu_layers,
        )
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_outputs import CausalLMOutputWithPast

        base_model = self.model.model if hasattr(self.model, "model") else self.model
        dev = self.compute_device
        compute_device = torch.device(dev)

        log_mem("start_forward")

        # embed + lm_head placement
        if not getattr(self, "_offloading_embed_on_device", False):
            base_model.embed_tokens.to(dev, non_blocking=True)
            base_model.rotary_emb.to(dev, non_blocking=True)
            self._offloading_embed_on_device = True
        if not getattr(self, "_offloading_lmhead_on_device", False):
            self.lm_head.to(dev, non_blocking=True)
            self._offloading_lmhead_on_device = True
        _load_layer_selective(base_model.layers[0], dev, non_blocking=True)
        torch.cuda.synchronize(compute_device)
        log_mem("after_embed+lmhead+layer0")

        # Input handling
        if input_ids is not None and input_ids.device != compute_device:
            input_ids = input_ids.to(compute_device, non_blocking=True)
        if inputs_embeds is None:
            inputs_embeds = base_model.embed_tokens(input_ids)
        if attention_mask is not None and attention_mask.device != inputs_embeds.device:
            attention_mask = attention_mask.to(inputs_embeds.device, non_blocking=True)
        if position_ids is not None and position_ids.device != inputs_embeds.device:
            position_ids = position_ids.to(inputs_embeds.device, non_blocking=True)
        if labels is not None and labels.device != inputs_embeds.device:
            labels = labels.to(inputs_embeds.device, non_blocking=True)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=base_model.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=base_model.config, input_embeds=inputs_embeds,
            attention_mask=attention_mask, cache_position=cache_position,
            past_key_values=past_key_values, position_ids=position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = base_model.rotary_emb(hidden_states, position_ids)

        log_mem("after_input_prep")

        # Layer loop
        load_stream = self.load_weight_stream
        unload_stream = self.unload_weight_stream
        num_layers = len(base_model.layers)

        with torch.cuda.stream(load_stream):
            _load_layer_selective(base_model.layers[0], dev, non_blocking=True)
        if num_layers > 1:
            with torch.cuda.stream(load_stream):
                _load_layer_selective(base_model.layers[1], dev, non_blocking=True)

        load_done_event = torch.cuda.Event()
        prev_compute_event = None
        _layers_pinned = getattr(self, '_offloading_layers_pinned', False)

        for i in range(num_layers):
            load_done_event.record(load_stream)
            torch.cuda.current_stream(compute_device).wait_event(load_done_event)

            hidden_states = base_model.layers[i](
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=past_key_values, cache_position=cache_position,
                position_embeddings=position_embeddings, use_cache=use_cache, **kwargs,
            )

            compute_event = torch.cuda.Event()
            compute_event.record(torch.cuda.current_stream(compute_device))

            if i + 2 < num_layers:
                with torch.cuda.stream(load_stream):
                    _load_layer_selective(base_model.layers[i + 2], dev, non_blocking=True)
            elif not getattr(self, '_norm_lmhead_prefetched', False):
                with torch.cuda.stream(load_stream):
                    base_model.norm.to(dev, non_blocking=True)
                self._norm_lmhead_prefetched = True

            if i >= 1:
                unload_stream.wait_event(prev_compute_event)
                _record_stream_for_module(base_model.layers[i - 1], unload_stream)
                _record_stream_for_weight_cache(base_model.layers[i - 1], unload_stream)
                _release_module_cached_weights(base_model.layers[i - 1])
                _clear_dequant_cache(base_model.layers[i - 1])
                with torch.cuda.stream(unload_stream):
                    if _layers_pinned:
                        _unload_weight_cache_to_pinned(base_model.layers[i - 1])
                        _unload_layer_to_pinned(base_model.layers[i - 1])
                    else:
                        _move_weight_cache_for_layer(base_model.layers[i - 1], "cpu", non_blocking=True)
                        base_model.layers[i - 1].to("cpu", non_blocking=True)

            prev_compute_event = compute_event

            if i in (0, 1, 2, 39, 40, 79):
                torch.cuda.synchronize(compute_device)
                log_mem(f"after_layer_{i}")

        # Finalize
        load_done_event.record(load_stream)
        torch.cuda.current_stream(compute_device).wait_event(load_done_event)
        torch.cuda.current_stream(compute_device).synchronize()

        hidden_states = base_model.norm(hidden_states)
        torch.cuda.synchronize(compute_device)
        log_mem("after_norm")

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # Unload last layer
        unload_stream.wait_event(prev_compute_event)
        _record_stream_for_module(base_model.layers[-1], unload_stream)
        _record_stream_for_weight_cache(base_model.layers[-1], unload_stream)
        _release_module_cached_weights(base_model.layers[-1])
        _clear_dequant_cache(base_model.layers[-1])
        with torch.cuda.stream(unload_stream):
            if _layers_pinned:
                _unload_weight_cache_to_pinned(base_model.layers[-1])
                _unload_layer_to_pinned(base_model.layers[-1])
            else:
                _move_weight_cache_for_layer(base_model.layers[-1], "cpu", non_blocking=True)
                base_model.layers[-1].to("cpu", non_blocking=True)

        # === THE BIG ONE: lm_head ===
        hs_for_lm = hidden_states[:, slice_indices, :]
        log_mem("before_lm_head")
        print(f"  >> hidden_states shape for lm_head: {hs_for_lm.shape}")
        print(f"  >> lm_head weight shape: {self.lm_head.weight.shape}")
        print(f"  >> expected logits: {hs_for_lm.shape[0]} x {hs_for_lm.shape[1]} x {self.lm_head.weight.shape[0]} "
              f"= {hs_for_lm.shape[0] * hs_for_lm.shape[1] * self.lm_head.weight.shape[0] * 2 / 1e9:.3f} GiB (bf16)")

        logits = self.lm_head(hs_for_lm)
        torch.cuda.synchronize(compute_device)
        log_mem("after_lm_head (LOGITS)")
        print(f"  >> logits shape: {logits.shape}, dtype: {logits.dtype}")
        print(f"  >> logits size: {logits.nelement() * logits.element_size() / 1e9:.3f} GiB")

        if not getattr(self, '_offloading_layers_pinned', False):
            torch.cuda.synchronize(compute_device)
            _pin_all_cpu_layers(base_model)
            self._offloading_layers_pinned = True

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    model.forward = MethodType(profiled_forward, model)

    # --- Run a single forward pass with realistic MMLU-like input ---
    # MMLU sequences are typically 256-512 tokens, batch_size=8
    seq_len = 512
    batch_size = 8
    print(f"\n{'='*80}")
    print(f"=== Profiling forward: batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*80}")

    torch.cuda.reset_peak_memory_stats(device)
    input_ids = torch.randint(0, 128256, (batch_size, seq_len), device=device)

    with torch.no_grad():
        output = model(input_ids)

    torch.cuda.synchronize(device)
    final_peak = torch.cuda.max_memory_allocated(device)
    final_alloc = torch.cuda.memory_allocated(device)
    print(f"\n{'='*80}")
    print(f"=== FINAL RESULTS ===")
    print(f"  Peak GPU memory: {final_peak/1e9:.3f} GiB")
    print(f"  Current allocated: {final_alloc/1e9:.3f} GiB")
    print(f"  Output logits shape: {output.logits.shape}")
    print(f"  Output logits size: {output.logits.nelement() * output.logits.element_size() / 1e9:.3f} GiB")
    print(f"{'='*80}")

    # Theoretical breakdown
    print(f"\n=== Theoretical Memory Breakdown ===")
    print(f"  embed_tokens:  {128256 * 8192 * 2 / 1e9:.3f} GiB")
    print(f"  lm_head:       {128256 * 8192 * 2 / 1e9:.3f} GiB")
    print(f"  norm:          {8192 * 2 / 1e6:.3f} MiB")
    print(f"  1 layer (bf16):{855654400 * 2 / 1e9:.3f} GiB")
    print(f"  2 layers:      {855654400 * 2 * 2 / 1e9:.3f} GiB")
    hs_bytes = batch_size * seq_len * 8192 * 2
    print(f"  hidden_states: {hs_bytes / 1e9:.3f} GiB")
    logits_bytes = batch_size * seq_len * 128256 * 2
    print(f"  logits tensor: {logits_bytes / 1e9:.3f} GiB")
    print(f"  causal_mask:   {batch_size * 1 * seq_len * seq_len * 4 / 1e9:.3f} GiB (float32)")

if __name__ == "__main__":
    profile_peak_memory()
