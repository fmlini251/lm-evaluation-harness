"""Ozaki-augmented HuggingFace model for lm_eval.

Usage:
    lm_eval --model ozaki-hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B,s_lst=2.3.4.6.12,k=256,dtype=bfloat16 \
        --tasks hellaswag

    # With offloading (loads model on CPU, layers streamed to GPU):
    lm_eval --model ozaki-hf \
        --model_args pretrained=meta-llama/Llama-3.1-8B,s_lst=2,k=256,dtype=bfloat16,offloading=True \
        --tasks hellaswag
"""

from __future__ import annotations

import json
import logging
import os
from types import MethodType

import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

eval_logger = logging.getLogger(__name__)


@register_model("ozaki-hf")
class OzakiHFLM(HFLM):
    """HFLM subclass that applies Ozaki custom matmul after model loading."""

    @staticmethod
    def _is_gptq_model(pretrained: str) -> bool:
        """Check if pretrained path contains a GPTQ quantized model."""
        config_path = os.path.join(pretrained, "config.json")
        if not os.path.isfile(config_path):
            return False
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            qconfig = config.get("quantization_config", {})
            return qconfig.get("quant_method", "") == "gptq"
        except (json.JSONDecodeError, OSError):
            return False

    def __init__(
        self,
        pretrained: str,
        # Ozaki-specific args (matching evaluate_ppl.py argparse)
        s_lst: str | None = None,
        k: int = 256,
        rslt_type: str = "ozaki",
        rounding: str = "round_half_away_from_0",
        scale_method: str = "new_compressed",
        shift_bits: int = 7,
        M_frac_bits: int = 8,
        # CustomGemmConfig args
        in_feature_ts: int = 4096,
        out_feature_ts: int = 4096,
        track_mtx_acc: bool = False,
        track_model_acc: bool = False,
        get_statistics: bool = False,
        # GlobalOzakiConfig args
        transfering_shift: bool = False,
        selective_s_boost: bool = False,
        target_patterns: str = "",
        # Offloading args
        offloading: bool = False,
        analyze_streams: bool = False,
        max_inflight_offloads: int = 2,
        # Weight caching
        weight_cache: bool = False,
        # Profiling
        profile: bool = False,
        # Pass remaining kwargs to HFLM
        **kwargs,
    ) -> None:
        # offloading=True -> load model on CPU, not GPU
        if offloading:
            kwargs["device"] = "cpu"

        # Auto-detect GPTQ quantized models and load via AutoGPTQ
        # so that layers become QuantLinear (required by ozaki_llama)
        if "autogptq" not in kwargs and "gptqmodel" not in kwargs:
            if self._is_gptq_model(pretrained):
                kwargs["autogptq"] = True
                eval_logger.info(f"Auto-detected GPTQ model at {pretrained}, using autogptq=True")

        super().__init__(pretrained=pretrained, **kwargs)

        # Force eager attention so create_causal_mask always produces a 4D mask.
        # AutoGPTQ ignores attn_implementation and defaults to "sdpa", which can
        # return None (relying on SDPA's is_causal flag). Our custom eager
        # attention needs an explicit mask, so without this fix the model runs
        # without causal masking and produces random-level accuracy.
        if hasattr(self.model, "config"):
            self.model.config._attn_implementation = "eager"

        if s_lst is None or s_lst == "":
            eval_logger.warning("No s_lst provided, skipping Ozaki matmul setup")
            return

        # Parse s_lst: "2.3.4.6.12" -> [2, 3, 4, 6, 12]
        parsed_s_lst = [int(s) for s in str(s_lst).split(".")]
        # Parse target_patterns: "attn_weights|mlp" -> ["attn_weights", "mlp"]
        parsed_target_patterns = [p for p in str(target_patterns).split("|") if p] if target_patterns else []
        k = int(k)
        in_feature_ts = int(in_feature_ts)
        out_feature_ts = int(out_feature_ts)
        shift_bits = int(shift_bits)
        M_frac_bits = int(M_frac_bits)
        max_inflight_offloads = int(max_inflight_offloads)

        from emulation.llm.ozaki_matmul import (
            CustomGemmConfig,
            GlobalOzakiConfig,
            reset_s_statistics,
            enable_profiling,
            reset_profiling_stats,
        )
        from emulation.llm.ozaki_llama import prepare_model_for_custom_matmul
        from emulation.llm.utils import prepare_ozaki, prepare_memory_offloading, offloading_forward

        # Auto-adjust tile sizes for 70B models (matching evaluate_ppl.py)
        if "70B" in str(pretrained):
            in_feature_ts = 14336
            out_feature_ts = 8192
            eval_logger.info("Detected 70B model, adjusted out_feature_ts=8192")

        custom_gemm_config = CustomGemmConfig(
            in_feature_ts=in_feature_ts,
            out_feature_ts=out_feature_ts,
            chunk_size=k,
            name="",
            track_mtx_acc=bool(track_mtx_acc),
            track_model_acc=bool(track_model_acc),
            get_statistics=bool(get_statistics),
            rslt_type=rslt_type,
        )

        log2M_tensor, moduli_tensor, invM_tensor, NMi_tensor, M_tensor = prepare_ozaki(parsed_s_lst, k)

        ozaki_config = GlobalOzakiConfig(
            s_lst=parsed_s_lst,
            log2M_tensor=log2M_tensor,
            moduli_tensor=moduli_tensor,
            NMi_tensor=NMi_tensor,
            M_tensor=M_tensor,
            invM_tensor=invM_tensor,
            rounding=rounding,
            scale_method=scale_method,
            transfering_shift=bool(transfering_shift),
            selective_s_boost=bool(selective_s_boost),
            target_patterns=parsed_target_patterns,
            shift_bits=shift_bits,
            M_frac_bits=M_frac_bits,
            weight_cache=bool(weight_cache),
        )

        # Offloading setup (matching evaluate_ppl.py: prepare_memory_offloading + offloading_forward)
        if offloading:
            compute_device = "cuda"
            prepare_memory_offloading(self.model, compute_device=compute_device)
            self.model.forward = MethodType(offloading_forward, self.model)
            # Override _device so lm_eval sends inputs to cuda
            self._device = torch.device(compute_device)
            eval_logger.info(f"[offloading] memory offloading enabled, compute_device={compute_device}")

            model = self.model
            if hasattr(model, 'max_inflight_unloads'):
                model.max_inflight_unloads = max(1, max_inflight_offloads)
                eval_logger.info(f"[offloading] max_inflight_unloads={model.max_inflight_unloads}")
            elif hasattr(model, 'model') and hasattr(model.model, 'max_inflight_unloads'):
                model.model.max_inflight_unloads = max(1, max_inflight_offloads)
                eval_logger.info(f"[offloading] max_inflight_unloads={model.model.max_inflight_unloads}")

        # Stream analysis setup
        if analyze_streams and offloading:
            model = self.model
            if hasattr(model, 'analyze_streams'):
                model.analyze_streams = True
                eval_logger.info("[analyze_streams] enabled")
            elif hasattr(model, 'model') and hasattr(model.model, 'analyze_streams'):
                model.model.analyze_streams = True
                eval_logger.info("[analyze_streams] enabled (ensemble base model)")

        # Profiling setup
        reset_s_statistics()
        if profile:
            reset_profiling_stats()
            enable_profiling(True)

        prepare_model_for_custom_matmul(
            model=self.model,
            custom_gemm_config=custom_gemm_config,
            ozaki_config=ozaki_config,
        )

        # ozaki_config to GPU regardless of offloading
        ozaki_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ozaki_config.to(ozaki_device)

        print(self.model)

        eval_logger.info(
            f"Ozaki applied: s_lst={parsed_s_lst}, k={k}, rslt_type={rslt_type}, "
            f"scale_method={scale_method}, shift_bits={shift_bits}, rounding={rounding}, "
            f"offloading={offloading}, weight_cache={weight_cache}"
        )
        eval_logger.info(f"CustomGemmConfig: {custom_gemm_config}")
        eval_logger.info(f"GlobalOzakiConfig: {ozaki_config}")
