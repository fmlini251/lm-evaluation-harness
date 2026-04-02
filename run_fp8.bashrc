#!/bin/bash

MODEL="neuralmagic/Meta-Llama-3.1-8B-FP8"

lm_eval \
--model ozaki-hf \
--model_args pretrained=$MODEL,dtype=float8,attn_implementation=eager,rslt_type=ozaki,s_lst=2,shift_bits=7,weight_cache=True,offloading=True \
--tasks mmlu \
--device cuda:0 \
--batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=$MODEL,dtype=float8,attn_implementation=eager,rslt_type=ozaki,s_lst=2,shift_bits=7,weight_cache=True,offloading=True \
--tasks winogrande \
--device cuda:0 \
--batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=$MODEL,dtype=float8,attn_implementation=eager,rslt_type=ozaki,s_lst=2,shift_bits=7,weight_cache=True,offloading=True \
--tasks arc_easy \
--device cuda:0 \
--batch_size 4

lm_eval \
--model ozaki-hf \
--model_args pretrained=$MODEL,dtype=float8,attn_implementation=eager,rslt_type=ozaki,s_lst=2,shift_bits=7,weight_cache=True,offloading=True \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 4
