#!/bin/bash

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,offloading=True,out_feature_ts=2048 \
# --tasks mmlu \
# --device cuda:0 \
# --batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,offloading=True,weight_cache=True,out_feature_ts=1024 \
--tasks winogrande \
--device cuda:0 \
--batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,offloading=True,weight_cache=True \
--tasks arc_easy \
--device cuda:0 \
--batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,offloading=True,weight_cache=True \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 4

# running

# lm_eval \
# --model ozaki-hf \
# --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,offloading=True,out_feature_ts=1024 \
# --tasks mmlu \
# --device cuda:0 \
# --batch_size 4 