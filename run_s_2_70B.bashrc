#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True,offloading=True,out_feature_ts=512,max_inflight_offloads=1 \
--tasks mmlu \
--device cuda:0 \
--batch_size 8

#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True,offloading=True,out_feature_ts=2048,max_inflight_offloads=1 \
--tasks winogrande \
--device cuda:0 \
--batch_size 8

#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True,offloading=True,out_feature_ts=2048,max_inflight_offloads=1 \
--tasks arc_easy \
--device cuda:0 \
--batch_size 8

#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True,offloading=True,out_feature_ts=2048,max_inflight_offloads=1 \
--tasks arc_challenge \
--device cuda:0 \
--batch_size 8

#!/bin/bash

lm_eval \
--model ozaki-hf \
--model_args pretrained=meta-llama/Llama-3.1-70B,dtype=bfloat16,attn_implementation=eager,rslt_type=ozaki,s_lst=2,weight_cache=True,offloading=True,out_feature_ts=2048,max_inflight_offloads=1 \
--tasks hellaswag \
--device cuda:0 \
--batch_size 8