#!/bin/bash
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=sdpa --tasks hellaswag --device cuda:0 --batch_size 16
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=sdpa --tasks arc_easy --device cuda:0 --batch_size 16
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=sdpa --tasks arc_challenge --device cuda:0 --batch_size 16
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=sdpa --tasks winogrande --device cuda:0 --batch_size 16
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B,dtype=bfloat16,attn_implementation=sdpa --tasks mmlu --device cuda:0 --batch_size 16
