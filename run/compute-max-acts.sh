#! /bin/bash
# For validating crosscoder
python scripts/compute_max_activation.py \
    --activation-cache-path $DATASTORE/activations \
    --dataset lmsys-chat-1m-chat-formatted \
    --model Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 \


python scripts/compute_max_activation.py \
    --activation-cache-path $DATASTORE/activations \
    --dataset fineweb-1m-sample \
    --model Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04 \
