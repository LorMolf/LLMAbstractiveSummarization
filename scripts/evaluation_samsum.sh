#/bin/bash


CACHE_DIR="cache"
OUT_DIR="results"

SEED=42

# Config
MAX_SAMPLES=10

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"

EVAL_BS=2

# Data
DATASET_NAME="samsum"
TEXT_COL="dialogue"
SUM_COL="summary"

MAX_TARGET_LENGTH=200

CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset_name $DATASET_NAME \
                                       --text_column $TEXT_COL \
                                       --summary_column $SUM_COL \
                                       --val_max_target_length $MAX_TARGET_LENGTH \
                                       --do_predict \
                                       --model_name_or_path $MODEL_NAME \
                                       --cache_dir $CACHE_DIR \
                                       --output_dir $OUT_DIR \
                                       --per_device_eval_batch_size $EVAL_BS \
                                       --pad_to_max_length \
                                       --fp16 \
                                       --seed $SEED \
                                       --max_predict_samples $MAX_SAMPLES
