#/bin/bash


CACHE_DIR="cache"
OUT_DIR="results"

SEED=42

# Config
MAX_TRAIN_SAMPLES=10
MAX_TEST_SAMPLES=5

TRAIN_BS=2
EVAL_BS=2

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
MODEL_NAME="NousResearch/Llama-2-7b-chat-hf"


# Data
DATASET_NAME="samsum"
TEXT_COL="dialogue"
SUM_COL="summary"


MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=200


CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset_name $DATASET_NAME \
                                       --text_column $TEXT_COL \
                                       --summary_column $SUM_COL \
                                       --val_max_target_length $MAX_TARGET_LENGTH \
                                       --max_source_length $MAX_SOURCE_LENGTH \
                                       --do_train \
                                       --do_eval \
                                       --do_predict \
                                       --model_name_or_path $MODEL_NAME \
                                       --cache_dir $CACHE_DIR \
                                       --output_dir $OUT_DIR \
                                       --per_device_train_batch_size $TRAIN_BS \
                                       --per_device_eval_batch_size $EVAL_BS \
                                       --pad_to_max_length \
                                       --seed $SEED \
                                       --max_train_samples $MAX_TRAIN_SAMPLES \
                                       --max_eval_samples $MAX_TEST_SAMPLES \
                                       --max_predict_samples $MAX_TEST_SAMPLES \
                                       --predict_with_generate \
                                       --use_peft \
                                       --fp16 \
