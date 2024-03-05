# LLMAbstractiveSummarization
Finetuning and evaluation of large language models in summarization tasks.

Training scripts:
- `main.py`: training with default Trainer;
- `main_sft.py`: (highly recommended) training using the SFTTrainer;

Utilities:
- `scr/data_classes.py`: list of arguments
- `src/prompts.py`: prompt formats specific for each model - change these configurations according to your model choice