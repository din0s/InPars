import argparse
import torch

import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

def split_triples(triples):
    examples = {
        'label': [],
        'text': [],
    }
    for i in range(len(triples['query'])):
        examples['text'].append(f'Query: {triples["query"][i]} Document: {triples["positive"][i]} Relevant:')
        examples['label'].append('true')
        examples['text'].append(f'Query: {triples["query"][i]} Document: {triples["negative"][i]} Relevant:')
        examples['label'].append('false')
    return examples

def tokenize(batch):
    tokenized = tokenizer(batch['text'])
    tokenized["labels"] = tokenizer(batch["label"])['input_ids']
    return tokenized

def get_lens(example):
    example["len"] = len(example["input_ids"])
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", type=str, required=True)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--base_model", type=str, default="castorini/monot5-base-msmarco-10k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    total_examples = None
    if args.max_steps:
        total_examples = args.gradient_accumulation_steps * args.per_device_train_batch_size * args.max_steps * torch.cuda.device_count()
        print(f"Training with {total_examples} examples!")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = load_dataset(
        'csv',
        data_files=args.triples,
        sep='\t',
        names=('query', 'positive', 'negative'),
    )
    dataset = dataset.map(
        split_triples,
        remove_columns=('query', 'positive', 'negative'),
        batched=True,
    )
    if total_examples:
        dataset['train'] = dataset['train'].shuffle().select(range(total_examples))

    dataset = dataset.map(tokenize, remove_columns=('text', 'label'), batched=True, desc='Tokenizing')
    lens = dataset['train'].map(get_lens, remove_columns=('input_ids', 'attention_mask', 'labels'))['len']
    m_len = np.mean(lens)
    s_len = np.std(lens)
    print(f"Mean length: {m_len}")
    print(f"SD length: {s_len}")

