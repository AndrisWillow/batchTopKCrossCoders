#! /bin/bash

set -x

SPLIT=train
BATCH_SIZE=4096 # 4096
# Model configuration
ACTIVATION_DIR="$DATASTORE/activations/"
LAYER=13
# Loading from buffer instead
BASE_MODEL="Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEVICE="cuda"
LR=5e-5
MU=0.041

# Parse command line arguments to check for custom mu value
custom_mu=false
for arg in "$@"; do
    if [[ $arg == --mu* ]]; then
        custom_mu=true
        break
    fi
done

# Build flags string
FLAGS="--activation-store-dir $ACTIVATION_DIR \
--batch-size $BATCH_SIZE \
--layer $LAYER \
--base-model $BASE_MODEL \
--chat-model $INSTRUCT_MODEL \
--same-init-for-all-layers \
--lr $LR \
--init-with-transpose \
--seed 42 \
--use-buffer \
--k 50 \
--num-samples 400_000_000
--num-validation-samples 2_000_000" # 400_000_000



# Only add default mu if not provided in command line arguments
if [ "$custom_mu" = false ]; then
    FLAGS="$FLAGS --mu $MU"
fi

additional_flags=$@

python scripts/train_crosscoder.py $FLAGS $additional_flags