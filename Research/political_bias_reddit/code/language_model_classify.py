import os, json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

ID2LABEL = {0: 'Alt', 1: 'Center', 2: 'Left', 3: 'Right'}
LABEL2ID = {'Alt': 0, 'Center': 1, 'Left': 2, 'Right': 3}

def load_data(data_dir, count_per_label, offset):
    """
    Loads text data from raw JSON files in the specified directory.

    Parameters:
    - data_dir (str): The directory containing raw JSON files.
    -count_per_label (int): number of data per label
    - offset (int): Offset to start reading data from   

    Returns:
    - data_dict (dict): A dictionary containing text and corresponding labels.
        - 'text': List of text bodies extracted from the JSON files.
        - 'labels': List of labels (converted to numerical ints) associated with each text entry.
    """

    data_dict = {'text': [], 'labels': []}

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            full_file = os.path.join(subdir, file)
            print(f"Processing {full_file}")
            with open(full_file) as open_file:
                data = json.load(open_file)
                
            start = offset % len(data)
            for i in range(start, start + count_per_label):
                safe_i = i % len(data)
                j = json.loads(data[safe_i])
                data_dict['text'].append(j['body'])
                data_dict['labels'].append(LABEL2ID[file]) 
                

    return data_dict

def tokenize_text(text, tokenizer):
    """
    Tokenizes a given text using the provided tokenizer.

    Parameters:
    - text (dict): The sample dict in which the text is to be tokenized.
    - tokenizer (object): The tokenizer object.
    
    Returns:
    - tokenized_text.
    """
    tokenized_text = tokenizer(text["text"], truncation=True, max_length=100)
    return tokenized_text

def compute_metrics(eval_pred):
    """
    Computes evaluation metrics for the model predictions.

    Parameters:
    - eval_pred (tuple): Tuple containing model predictions and true labels.

    Returns:
    - Computed metrics based on model predictions and true labels.
    """
    predictions, labels = eval_pred
    return {
        "accuracy": np.mean(np.argmax(predictions, axis=1) == labels)
    }

def train_model(model, tokenizer, tokenized_dataset, data_collator, training_args):
    """
    Trains the model using the provided Trainer object.

    Parameters:
    - model (object): The model to be trained.
    - tokenizer (object): The tokenizer object.
    - tokenized_dataset (object): Tokenized dataset.
    - data_collator (object): Data collator with padding.
    - training_args (object): Training arguments for the Trainer.
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    evaluation_result = trainer.evaluate()

    return evaluation_result

def main(args):
    """
    Main function that orchestrates the training process.
    """

    if args.toy:
        count_per_label = 10
    else:
        count_per_label = 10000
        assert torch.cuda.is_available(), "Need cuda to train the model on non-toy dataset."

    data_dict = load_data(args.input_dir, count_per_label, args.offset)
    dataset = Dataset.from_dict(data_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        "launch/POLITICS", 
        cache_dir=args.cache_dir
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        "launch/POLITICS", 
        cache_dir=args.cache_dir, 
        num_labels=len(ID2LABEL)
        )

    # Set the device to CUDA if available, otherwise to CPU, and move the model to the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use the tokenize_text function to tokenize the dataset. Consider using batched processing for efficiency.
    tokenized_dataset = dataset.map(
        lambda x: tokenize_text(x, tokenizer),
        batched=True
    )
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    # Set the TrainingArguments for training the model.
    training_args = TrainingArguments(
        report_to="none",
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    evaluation_result = train_model(model, tokenizer, tokenized_dataset, data_collator, training_args)

    output_file = args.output_dir / "evaluation_result.txt"

    with open(output_file, "w") as f:
        json.dump(evaluation_result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0,
        help="Offset to start reading data from.")
    parser.add_argument("-i", "--input-dir",
        help="Specify the input directory of data.",
        type=Path,
        required=True)
    parser.add_argument(
        "-o", "--output-dir",
        help="Specify the directory to write the training results.",
        type=Path,
        default=Path("finetuned_results"))
    parser.add_argument(
        "--cache-dir",
        help="Specify the directory to cache the POLITICS model.",
        type=Path,
        required=True)
    parser.add_argument('--toy', action="store_true",
        help="Train the model on the toy dataset.")
    args = parser.parse_args()

    main(args)
