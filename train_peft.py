import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

from helpers import datasets
from TextWiz.textwiz import loader, get_empty_conversation_template

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
reverse_mapping = {v:k for k,v in loader.ALL_MODELS_MAPPING.items()}
textwiz_model_name = reverse_mapping[model_name]

result_folder = 'training_results_peft'

training_args = TrainingArguments(
        optim='adamw_torch',
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=5,
        bf16=True,
        dataloader_num_workers=2,
        output_dir=result_folder,
        logging_dir=result_folder + '/logs',
        logging_strategy='epoch',
        report_to='tensorboard',
        save_strategy='epoch',
        save_only_model=True,
        seed=123,
    )

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)


def main():

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # attn_implementation='flash_attention_2',
        attn_implementation='sdpa',
        torch_dtype=loader.get_model_dtype(textwiz_model_name),
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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


def merge_all_model_checkpoints():

    # Load the tokenizer to save it in the same destination along the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    checkpoints = [os.path.join(result_folder, folder) for folder in os.listdir(result_folder) if folder != 'logs' and not folder.startswith('.')]
    destinations = [os.path.join(result_folder, 'merged_models', folder) for folder in os.listdir(result_folder) if folder != 'logs' and not folder.startswith('.')]
    # Fully merge the Lora weights and resave model
    for checkpoint, destination in zip(checkpoints, destinations):
        lora_model = AutoPeftModelForCausalLM.from_pretrained(checkpoint, torch_dtype=loader.get_model_dtype(textwiz_model_name)).cuda()
        lora_model = lora_model.merge_and_unload()
        lora_model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
        del lora_model


if __name__ == '__main__':

    # main()
    merge_all_model_checkpoints()