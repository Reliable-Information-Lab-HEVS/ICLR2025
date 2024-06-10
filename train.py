import os

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from helpers import datasets
from TextWiz.textwiz import loader

def main():

    training_args = TrainingArguments(
        optim='adamw_torch',
        per_device_train_batch_size=3,
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True,
        dataloader_num_workers=4,
        torch_compile=True,
        output_dir='training_results',
        logging_strategy='epoch',
        save_strategy='epoch',
        report_to='tensorboard',
        seed=123,
        local_rank=int(os.environ["LOCAL_RANK"]),
    )

    model_name = 'llama3-8B-instruct'
    model, tokenizer = loader.load_model_and_tokenizer(model_name)

    dataset = datasets.WalliserDeutschDataset(tokenizer, sample_size=loader.get_model_context_size(model_name))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == '__main__':

    main()