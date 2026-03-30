#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks mmlu \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True \
--tasks winogrande \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks arc_easy \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16


lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks mmlu \
--device cuda:0 \
--batch_size 8

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks winogrande \
--device cuda:0 \
--batch_size 8

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks arc_easy \
--device cuda:0 \
--batch_size 8

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 8

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemm_16,s_lst=2,offloading=True,out_feature_ts=4096 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 8

