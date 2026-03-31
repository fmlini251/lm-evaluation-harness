#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=/home/howonlee/Llama-3.1-8B-8bit-256g-fp16,dtype=float16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True \
--tasks mmlu \
--device cuda:0 \
--batch_size 2

lm_eval \
--model ozaki-hf \
--model_args pretrained=/home/howonlee/Llama-3.1-8B-8bit-256g-fp16,dtype=float16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True \
--tasks winogrande \
--device cuda:0 \
--batch_size 2

lm_eval \
--model ozaki-hf \
--model_args pretrained=/home/howonlee/Llama-3.1-8B-8bit-256g-fp16,dtype=float16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True \
--tasks arc_easy \
--device cuda:0 \
--batch_size 2

lm_eval \
--model ozaki-hf \
--model_args pretrained=/home/howonlee/Llama-3.1-8B-8bit-256g-fp16,dtype=float16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 2

lm_eval \
--model ozaki-hf \
--model_args pretrained=/home/howonlee/Llama-3.1-8B-8bit-256g-fp16,dtype=float16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True \
--tasks hellaswag \
--device cuda:0 \
--batch_size 2

