#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=4 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=4 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=3,out_feature_ts=2048 \
--tasks mmlu \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=3 \
--tasks winogrande \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=3 \
--tasks arc_easy \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=3 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=3 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16


lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=2,out_feature_ts=2048 \
--tasks mmlu \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=2 \
--tasks winogrande \
--device cuda:0 \
--batch_size 16
lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=2 \
--tasks arc_easy \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=2 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=gemmul8_accurate,s_lst=2 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16
