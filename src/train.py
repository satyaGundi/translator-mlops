import mlflow
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)

SOURCE_LANG = "en"
TARGET_LANG = "es"

def main():
    # Choose small model
    model_name = "Helsinki-NLP/opus-mt-en-es"
    mlflow.set_experiment("translator-en-es")

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)

        # Load dataset (tiny subset for speed)
        full_ds = load_dataset("opus_books", "en-es", split="train[:2500]")
        train_ds = full_ds.select(range(2000))
        eval_ds = full_ds.select(range(2000, 2500))

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, from_safetensors=True)

        def preprocess(batch):
            translations = batch["translation"]
            src_texts = [ex.get(SOURCE_LANG) for ex in translations]
            tgt_texts = [ex.get(TARGET_LANG) for ex in translations]

            model_inputs = tokenizer(src_texts, truncation=True, max_length=128)
            # Newer API: pass targets via text_target
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(text_target=tgt_texts, truncation=True, max_length=128)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        remove_cols = full_ds.column_names  # usually ["translation"]
        train_ds = train_ds.map(preprocess, batched=True, remove_columns=remove_cols)
        eval_ds  = eval_ds.map(preprocess,  batched=True, remove_columns=remove_cols)

        # Model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        metric = evaluate.load("sacrebleu")

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_labels = [[l] for l in decoded_labels]
            return {"sacrebleu": metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]}

        training_args = TrainingArguments(
            output_dir="outputs",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            eval_strategy="epoch",
            logging_steps=20,
            predict_with_generate=True,
            save_strategy="no"  # keep it light
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()