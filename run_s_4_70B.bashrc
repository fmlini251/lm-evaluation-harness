#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=4,weight_cache=True,offloading=True \
--tasks mmlu \
--device cuda:0 \
--batch_size 8