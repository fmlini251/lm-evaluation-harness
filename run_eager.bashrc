#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,offloading=True \
--tasks arc-easy \
--device cuda:0 \
--batch_size 8

