#!/bin/bash

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2 \
# --tasks piqa \
# --device cuda:0 \
# --batch_size 16

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2 \
# --tasks openbookqa \
# --device cuda:0 \
# --batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,out_feature_ts=1024 \
--tasks piqa \
--device cuda:0 \
--batch_size 16

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4 \
# --tasks openbookqa \
# --device cuda:0 \
# --batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3,out_feature_ts=2048 \
--tasks piqa \
--device cuda:0 \
--batch_size 16

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3 \
# --tasks openbookqa \
# --device cuda:0 \
# --batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,out_feature_ts=2048 \
--tasks piqa \
--device cuda:0 \
--batch_size 16

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2 \
# --tasks openbookqa \
# --device cuda:0 \
# --batch_size 16