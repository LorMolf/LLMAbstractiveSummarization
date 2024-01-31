#!/bin/bash


DATASET_NAME="billsum"

TEXT_COL="text"
SUM_COL="summary"

CACHE_DIR="cache"
OUT_DIR="results"

TRAIN_BS=1
EVAL_BS=1

MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=200

LR='1e-4'

# GEN PARAMS
TOP_K=30
TOP_P=0.9
TEMPERATURE=0.8

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
model_type='zephyr'

NEFTUNE=5
WARMUP_STEPS=25





for SEED in 42
do
    for FS in 10 100
    do
        MAX_STEPS=$(($FS*10))
        
        CUDA_VISIBLE_DEVICES=0 python3 main_sft.py --dataset_name $DATASET_NAME \
                                            --text_column $TEXT_COL \
                                            --summary_column $SUM_COL \
                                            --val_max_target_length $MAX_TARGET_LENGTH \
                                            --max_source_length $MAX_SOURCE_LENGTH \
                                            --model_name_or_path $MODEL_NAME \
                                            --cache_dir $CACHE_DIR \
                                            --output_dir $OUT_DIR \
                                            --per_device_train_batch_size $TRAIN_BS \
                                            --per_device_eval_batch_size $EVAL_BS \
                                            --pad_to_max_length \
                                            --seed $SEED \
                                            --max_train_samples $FS \
                                            --use_peft \
                                            --fp16 \
                                            --max_steps $MAX_STEPS \
                                            --warmup_steps $WARMUP_STEPS \
                                            --neftune_noise_alpha $NEFTUNE\
                                            --optim "paged_adamw_8bit" \
                                            --lr_scheduler_type "cosine" \
                                            --learning_rate $LR \
                                            --do_predict \
                                            --do_train \
                                            --report_to "wandb" \
                                            --save_strategy "no" \
                                            --save_strategy "no"
    done
done



MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
model_type="llama2"

NEFTUNE=15
WARMUP_STEPS=20

for SEED in 42
do
    for FS in 10 100
    do
        MAX_STEPS=$(($FS*10))
        
        CUDA_VISIBLE_DEVICES=0 python3 main_sft.py --dataset_name $DATASET_NAME \
                                            --text_column $TEXT_COL \
                                            --summary_column $SUM_COL \
                                            --val_max_target_length $MAX_TARGET_LENGTH \
                                            --max_source_length $MAX_SOURCE_LENGTH \
                                            --model_name_or_path $MODEL_NAME \
                                            --cache_dir $CACHE_DIR \
                                            --output_dir $OUT_DIR \
                                            --per_device_train_batch_size $TRAIN_BS \
                                            --per_device_eval_batch_size $EVAL_BS \
                                            --pad_to_max_length \
                                            --seed $SEED \
                                            --max_train_samples $FS \
                                            --use_peft \
                                            --fp16 \
                                            --max_steps $MAX_STEPS \
                                            --warmup_steps $WARMUP_STEPS \
                                            --neftune_noise_alpha $NEFTUNE\
                                            --optim "paged_adamw_8bit" \
                                            --lr_scheduler_type "cosine" \
                                            --learning_rate $LR \
                                            --do_predict \
                                            --do_train \
                                            --report_to "wandb" \
                                            --save_strategy "no"
    done
done