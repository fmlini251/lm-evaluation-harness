#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks mmlu \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks winogrande \
--device cuda:0 \
--batch_size 16
lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks arc_easy \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 16

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks hellaswag \
--device cuda:0 \
--batch_size 16

