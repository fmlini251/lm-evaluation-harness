#!/bin/bash

# Batch-level checkpoint/resume for all benchmarks.
# If the server crashes, just re-run — resumes from last completed batch.
#
# === Single GPU ===
#   Just run this script.
#
# === Multi-GPU (range split) ===
#   GPU 0: python run_with_checkpoint.py ... --device cuda:0 --batch_range 1:300
#   GPU 1: python run_with_checkpoint.py ... --device cuda:1 --batch_range 301:700
#   GPU 2: python run_with_checkpoint.py ... --device cuda:2 --batch_range 701:1000
#   Merge: python run_with_checkpoint.py ... --merge
#   Final: python run_with_checkpoint.py ... (no --batch_range, loads all from cache)

python run_with_checkpoint.py \
    --model ozaki-hf \
    --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3,weight_cache=True,offloading=True,out_feature_ts=512 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

python run_with_checkpoint.py \
    --model ozaki-hf \
    --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3,weight_cache=True,offloading=True \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

python run_with_checkpoint.py \
    --model ozaki-hf \
    --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3,weight_cache=True,offloading=True \
    --tasks arc_challenge \
    --device cuda:0 \
    --batch_size 8

python run_with_checkpoint.py \
    --model ozaki-hf \
    --model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=3,weight_cache=True,offloading=True \
    --tasks winogrande \
    --device cuda:0 \
    --batch_size 8
