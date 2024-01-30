import logging
import os
import math
import torch
import sys
import pandas as pd
from tqdm import tqdm

import wandb

import datasets
import evaluate
import nltk
import numpy as np
from datasets import load_dataset, load_from_disk
from filelock import FileLock
from torch.utils.data import DataLoader


import transformers

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
    TaskType
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer, 
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    set_seed,
    get_scheduler
)


from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from src.data_classes import ModelArguments, DataTrainingArguments

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


INSTRUCTION_PROMPT = """
You are an assistant capable of producing faithful and concise summaries of an input document. 
Read the text provided by the user and summarize it by keeping the most useful information which
you consider to best sum up the content of the document. Be as concise as needed and do not
include information out of the input text's domain.\n"""
USER_PROMPT = 'Summarize the following text:\n'

PROMPTS = {
    'zephyr' : {
        'instruction' : f'<|system|> {INSTRUCTION_PROMPT}',
        'user' : f'<|user|> {USER_PROMPT}',
        'answer' : '<|assistant|>'
    },
    # 'llama2' : {
    #     'instruction' : f'[INST] <<SYS>> {INSTRUCTION_PROMPT}',
    #     'user' : f'<</SYS>> {USER_PROMPT}',
    #     'answer' : '[/INST]'
    # }
    'llama2' : {
        'instruction' : f'# Assistant:\n {INSTRUCTION_PROMPT}',
        'user' : f'# Summarize:\n {USER_PROMPT}',
        'answer' : '# Summary:'
    }
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir += "/" + model_args.model_name_or_path.split("/")[-1] + '/' + '_'.join([data_args.dataset_name, str(data_args.max_train_samples), str(training_args.seed)])

    assert not os.path.exists(training_args.output_dir), "Output directory already exists"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name_local is not None:
        # Loading a local dataset.
        raw_datasets = load_from_disk(data_args.dataset_name_local)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.lang,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Shuffle dataset
    raw_datasets = raw_datasets.shuffle(seed=training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if model_args.use_peft:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                     quantization_config=bnb_config)
        
        # prepare int-8 model for training
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM")

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float16 if training_args.fp16 else torch.float32)

    model.config.use_cache = False



    if training_args.report_to == 'wandb':
        wandb.init(
            project='Few-shot-Summarization',
            name=f"{model_args.model_type}-{data_args.dataset_name}", 
            config={
                "architecture": model_args.model_type,
                "dataset": data_args.dataset_name,
                "n_train_data": data_args.max_train_samples,
            }
        )

        #wandb.watch(model, log_freq=100)


    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    
    if tokenizer.pad_token is None:
      print("No padding token - using EOS instead")
      tokenizer.pad_token = tokenizer.eos_token
    
    
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else "Write a summary of the following text.\n"

    # Preprocessing the datasets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    text_column = data_args.text_column
    if text_column is None or text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be defined as one of: {', '.join(column_names)}"
        )
    summary_column = data_args.summary_column
    if summary_column is None or text_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{data_args.summary_column}' needs to be defined as one of: {', '.join(column_names)}"
        )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    model_type = model_args.model_type
    INSTR = PROMPTS[model_type]['instruction']
    USER = PROMPTS[model_type]['user']
    ANSWER = PROMPTS[model_type]['answer']

    # preprocessing
    def format_dataset(sample):
        documents = sample[text_column]
        summaries = sample[summary_column]

        inputs = []    

        # append prefix and postfix prompts
        instr = list(map(lambda x: ' '.join([INSTR, x, USER]), documents))
        resp = list(map(lambda x: ' '.join([ANSWER,x]), summaries))
        inputs = list(map(lambda x: '\n'.join([x[0], x[1], tokenizer.eos_token ]),zip(instr, resp)))

        return inputs

    # template dataset to add prompt to each sample
    def preprocess_function(sample):
        out = {}
        out["text"] = format_dataset(sample)
        return out

    def preprocess_function_w_tok(examples):
        model_inputs = tokenizer(examples['text'], 
                                 max_length=data_args.max_source_length, 
                                 padding=padding, 
                                 truncation=True,
                                 return_tensors='pt')
        return model_inputs

    # Data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ANSWER, 
        tokenizer=tokenizer, 
        mlm=False)

    if training_args.do_train:
        train_dataset = raw_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator,
            batch_size=training_args.per_device_train_batch_size
        )
        

        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch


        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        if not model_args.use_peft:
            no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=max_train_steps
            )
        else:
            optimizer = 'paged_adamw_8bit'
            lr_scheduler = 'cosine'
        
        optimizers = (optimizer, lr_scheduler)
    else:
        optimizers = (None, None)

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
            
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Metric
    metric_rouge = evaluate.load("rouge")

    def __compute_rouge(decoded_preds, decoded_labels):

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]
        
        result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 2) for k, v in result.items()}

        result["R"] = round(np.mean([result["rouge1"], result["rouge2"], result["rougeL"]]) / \
            (1 + (np.var([result["rouge1"]/100, result["rouge2"]/100, result["rougeL"]/100]))), 2)

        decoded_preds = [pred.replace("\n", " ") for pred in decoded_preds]
        decoded_labels = [label.replace("\n", " ") for label in decoded_labels]

        return result

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.argmax(preds,axis=-1)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = [pred.split(ANSWER)[-1]\
                             .replace('</s>', '')\
                             .replace('<s>','')\
                             .strip() for pred in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [label.split(ANSWER)[-1]\
                               .replace('</s>', '')\
                               .replace('<s>','')\
                               .strip() for label in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        result = __compute_rouge(decoded_preds, decoded_labels)

        result["gen_len"] = np.mean([len(pred) for pred in decoded_preds])

        return result

    trainer = None

    # Training
    if training_args.do_train:

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=data_args.max_source_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            compute_metrics=compute_metrics
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train()#resume_from_checkpoint=checkpoint)
        #trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        #trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if trainer is None:
            trainer = SFTTrainer(
                model=model,
                eval_dataset=eval_dataset,
                peft_config=lora_config,
                dataset_text_field="text",
                max_seq_length=data_args.max_source_length,
                tokenizer=tokenizer,
                args=training_args,
                packing=False,
                compute_metrics=compute_metrics
            )


        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # reinstantiate the trainer with the prediction dataset as validation
        model.eval()
        trainer = SFTTrainer(
            model=model,
            eval_dataset=predict_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=data_args.max_source_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
            compute_metrics=compute_metrics
        )
        
        metrics = trainer.evaluate(metric_key_prefix="predict")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        
        if trainer.is_world_process_zero():
            if model_args.predict_with_generate:
                
                model = model.merge_and_unload()

                # tokenize data
                generation_ds = predict_dataset.map(preprocess_function_w_tok,
                                                    batched=True,
                                                    remove_columns=['text'])

                generation_dataloader = DataLoader(generation_ds, 
                                                    batch_size=training_args.per_device_eval_batch_size,
                                                    collate_fn=transformers.DefaultDataCollator())


                raw_test_labels = raw_datasets['test'].select(range(max_predict_samples))
                def get_summaries(x):
                    out={}
                    out[summary_column] = x[summary_column]
                    return out
                raw_test_labels = raw_test_labels.map(get_summaries, batched=True, remove_columns=raw_test_labels.column_names)

                labels_dataloader = DataLoader(raw_test_labels, 
                                                batch_size=training_args.per_device_eval_batch_size)
                gen_config = {
                    'temperature' : model_args.temperature,
                    'do_sample' : model_args.do_sample,
                    'top_k' : model_args.top_k,
                    'top_p' : model_args.top_p
                }

                prediction_lst = []
                metrics_tot = None

                for gen_data_split, label_split in tqdm(zip(generation_dataloader, labels_dataloader), desc='Generating answers ...', total=len(generation_dataloader)):

                    gen_data_concat = {key: torch.cat([v.view(1,data_args.max_source_length) for v in value], dim=0).to(model.device) for key, value in gen_data_split.items()}
                    #print(f"<>>>>> {gen_data_concat['input_ids'].shape} | {gen_data_split['input_ids'][0].shape}")
                    predictions = model.generate(**gen_data_concat, **gen_config)
                    
                    predictions = tokenizer.batch_decode(
                        predictions, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.split(ANSWER)[-1]\
                                        .replace('</s>', '')\
                                        .replace('<s>','')\
                                        .strip() for pred in predictions]

                    labels = [label.split(ANSWER)[-1]\
                                    .replace('</s>', '')\
                                    .replace('<s>','')\
                                    .strip() for label in label_split[summary_column]]


                    metrics = __compute_rouge(predictions, labels)

                    metrics["gen_len"] = np.mean([len(pred) for pred in predictions])


                    if metrics_tot is None:
                        metrics_tot = metrics
                    else:
                        for k in metrics_tot: metrics_tot[k] += metrics[k]

                    prediction_lst += predictions

                for k in metrics_tot: metrics_tot[k] /= max_predict_samples

                gen_metrics = {}
                for k in metrics_tot: gen_metrics[f'generation_{k}'] = metrics_tot[k]

                trainer.log_metrics("generation", gen_metrics)
                trainer.save_metrics("generation", gen_metrics)
                
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(prediction_lst))
        
    kwargs = {"finetuned_from": model_args.model_name_or_path}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    wandb.finish()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()