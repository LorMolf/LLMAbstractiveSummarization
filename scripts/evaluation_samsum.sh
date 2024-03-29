#/bin/bash


CACHE_DIR="cache"
OUT_DIR="results"

SEED=42

# Config
MAX_TEST_SAMPLES=10

MAX_TRAIN_SAMPLES=100
MAX_STEPS=1000
WARMUP_STEPS=25


TRAIN_BS=1
EVAL_BS=4

MODEL_NAME="HuggingFaceH4/zephyr-7b-beta"
model_type='zephyr'

# MODEL_NAME="meta-llama/Llama-2-13b-chat-hf"
# model_type="llama2"

# MODEL_NAME='microsoft/phi-2'
# model_type='phi-2'


# Data
DATASET_NAME="samsum"
TEXT_COL="dialogue"
SUM_COL="summary"




MAX_SOURCE_LENGTH=1024
MAX_TARGET_LENGTH=50

LR='1e-4'

# GEN PARAMS
TOP_K=30
TOP_P=0.9
TEMPERATURE=0.8


#  python3 main.py 
python3 main_sft.py --dataset_name $DATASET_NAME \
                     --text_column $TEXT_COL \
                     --summary_column $SUM_COL \
                     --val_max_target_length $MAX_TARGET_LENGTH \
                     --max_source_length $MAX_SOURCE_LENGTH \
                     --model_name_or_path $MODEL_NAME \
                     --pad_to_max_length \
                     --cache_dir $CACHE_DIR \
                     --output_dir $OUT_DIR \
                     --per_device_train_batch_size $TRAIN_BS \
                     --per_device_eval_batch_size $EVAL_BS \
                     --max_train_samples $MAX_TRAIN_SAMPLES \
                     --max_predict_samples 10 \
                     --max_steps $MAX_STEPS \
                     --warmup_steps $WARMUP_STEPS \
                     --neftune_noise_alpha '5'\
                     --optim " paged_adamw_32bit" \
                     --lr_scheduler_type "cosine" \
                     --learning_rate $LR \
                     --seed $SEED \
                     --use_peft \
                     --fp16 \
                     --report_to "wandb" \
                     --do_predict \
                     --save_strategy "no" \
                     --save_predictions_to_file \
                     --do_sample \
                     --top_k $TOP_K \
                     --top_p $TOP_P \
                     --temperature $TEMPERATURE \
                     --eval_accumulation_steps 30 \
                     --max_predict_samples $MAX_TEST_SAMPLES \
                     --max_eval_samples $MAX_TEST_SAMPLES \
                     --do_eval \
                     --predict_with_generate
                                       