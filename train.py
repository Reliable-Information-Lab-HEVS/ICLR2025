import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.training_args import OptimizerNames

from helpers import datasets
from TextWiz.textwiz import loader, get_empty_conversation_template

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
reverse_mapping = {v:k for k,v in loader.ALL_MODELS_MAPPING.items()}
textwiz_model_name = reverse_mapping[model_name]

result_folder = 'training_results'

training_args = TrainingArguments(
        optim='adamw_torch',
        # optim=OptimizerNames.SGD,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True,
        dataloader_num_workers=2,
        # torch_compile=True,
        output_dir=result_folder,
        logging_dir=result_folder + '/logs',
        logging_strategy='epoch',
        report_to='tensorboard',
        save_strategy='epoch',
        save_only_model=True,
        seed=123,
    )


def main():

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation='flash_attention_2',
        torch_dtype=loader.get_model_dtype(textwiz_model_name),
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = datasets.WalliserDeutschDataset(tokenizer, template=get_empty_conversation_template(textwiz_model_name),
                                              sample_size=loader.get_model_context_size(textwiz_model_name))
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