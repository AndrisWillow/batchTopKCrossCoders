#! /bin/bash

set -x

# Define datasets and other constants
# TODO make our own defintions; Maybe just validate with chat model; or make smaller validation?
CHAT_DATASET=AndrisWillow/lmsys-500k-Llama3.2_chat_format
FINEWEB_DATASET=AndrisWillow/pile-500k
ACTIVATION_STORE_DIR=/workspace/data/activations
BATCH_SIZE=32 # RTX3090 can't handle more than this
CHAT_MODEL=meta-llama/Llama-3.2-1B-Instruct
BASE_MODEL=meta-llama/Llama-3.2-1B
TEXT_COLUMN=text

# Initialize variables for command-line arguments
SPLIT_ARG=""
DATASET_ARG=""

# Parse the command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --split)
            SPLIT_ARG="$2"
            shift 2
            ;;
        --dataset)
            DATASET_ARG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --split <train|val> --dataset <chat|fineweb>"
            exit 1
            ;;
    esac
done

# Validate that both arguments were supplied
if [ -z "$SPLIT_ARG" ] || [ -z "$DATASET_ARG" ]; then
    echo "Usage: $0 --split <train|val> --dataset <chat|fineweb>"
    exit 1
fi

# Validate split argument and set corresponding values
if [ "$SPLIT_ARG" == "train" ]; then
    SPLIT="train"
    N_TOKS=50_000_000
    if [ "$DATASET_ARG" == "chat" ]; then
        N_TOKS=100_000_000
    fi
elif [ "$SPLIT_ARG" == "val" ]; then
    SPLIT="validation"
<<<<<<< HEAD
    N_TOKS=5_000_000
=======
    N_TOKS=2_500_000
>>>>>>> ad87a96f7cf12c20c3d6dc4c5863e268b8fefa13
else
    echo "Error: --split must be either 'train' or 'val'"
    exit 1
fi

# Validate dataset argument and choose dataset accordingly
if [ "$DATASET_ARG" == "chat" ]; then
    DATASET=$CHAT_DATASET
elif [ "$DATASET_ARG" == "fineweb" ]; then
    DATASET=$FINEWEB_DATASET
else
    echo "Error: --dataset must be either 'chat' or 'fineweb'"
    exit 1
fi

# Build common flags using the updated variables
COMMON_FLAGS="--dtype float32 \
--disable-multiprocessing \
--store-tokens \
--text-column $TEXT_COLUMN \
--batch-size $BATCH_SIZE \
--layers 13 \
--dataset $DATASET \
--dataset-split $SPLIT \
--activation-store-dir $ACTIVATION_STORE_DIR \
--max-tokens $N_TOKS"

# Run activation collection for both base and chat models
python scripts/collect_activations.py $COMMON_FLAGS --model $BASE_MODEL
python scripts/collect_activations.py $COMMON_FLAGS --model $CHAT_MODEL

# ./run/compute-activations-AndrisW --split <train|val> --dataset <chat|fineweb>