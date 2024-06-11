import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from helpers import datasets
from TextWiz.textwiz import loader

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
reverse_mapping = {v:k for k,v in loader.ALL_MODELS_MAPPING.items()}
textwiz_model_name = reverse_mapping[model_name]

training_args = TrainingArguments(
        # optim='adamw_torch',
        optim='adafactor',
        per_device_train_batch_size=3,
        learning_rate=5e-5,
        num_train_epochs=5,
        # bf16=True,
        dataloader_num_workers=2,
        # torch_compile=True,
        # torch_compile_backend=
        output_dir='training_results',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='tensorboard',
        seed=123,
    )


def main():

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation='flash_attention_2',
        torch_dtype=loader.get_model_dtype(textwiz_model_name),
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = datasets.WalliserDeutschDataset(tokenizer, sample_size=loader.get_model_context_size(textwiz_model_name))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer
    )

    training_results = trainer.train()
    # print(training_results)


if __name__ == '__main__':

    main()