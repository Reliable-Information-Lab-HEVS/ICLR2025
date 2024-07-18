from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model

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
        attn_implementation='flash_attention_2',
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


if __name__ == '__main__':

    main()