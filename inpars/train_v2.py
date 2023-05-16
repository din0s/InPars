from dataclasses import dataclass, field
from typing import Literal

import torch
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class ExtraArguments:
    triples: str = field(
        metadata={
            "help": "Triples file containing query, positive and negative examples (TSV format)."
        },
    )
    base_model: str = field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        metadata={"help": "Base model to fine-tune."},
    )
    model_type: Literal["bert", "t5"] = field(
        default="bert",
        metadata={"help": "Model type to fine-tune."},
    )
    max_doc_length: int = field(
        default=300,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )


def infer_type(triple):
    if "_" in triple["id"]:
        triple["type"] = "keyword"
    elif int(triple["id"]) < 819:
        triple["type"] = "acronym"
    else:
        triple["type"] = "inpars"
    return triple


def split_triples_bert(triples):
    examples = {
        "queries": [],
        "docs": [],
        "label": [],
    }
    for i in range(len(triples["query"])):
        examples["queries"].extend([triples["query"][i]] * 2)
        examples["docs"].extend(
            [
                str(triples["positive"][i]).strip(),
                str(triples["negative"][i]).strip(),
            ]
        )
        examples["label"].extend([1.0, 0.0])
    return examples

def split_triples_t5(triples):
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


def tokenize_bert(batch):
    tokenized = tokenizer(
        batch["queries"],
        batch["docs"],
        padding=True,
        truncation="longest_first",
        max_length=args.max_doc_length,
    )
    return tokenized


def tokenize_t5(batch):
    tokenized = tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=args.max_doc_length,
    )
    tokenized["labels"] = tokenizer(batch["label"])['input_ids']
    return tokenized


if __name__ == "__main__":
    args = HfArgumentParser(ExtraArguments).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    train_args_class = Seq2SeqTrainingArguments if args.model_type == "t5" else TrainingArguments
    parser = HfArgumentParser((train_args_class, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "no"
    training_args.do_eval = False
    set_seed(training_args.seed)

    total_examples = None
    if training_args.max_steps > 0:
        total_examples = (
            training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
            * training_args.max_steps
            * max(1, torch.cuda.device_count())
            // 2  # split into (query, pos), (query, neg)
        )

    model_class = (
        AutoModelForSeq2SeqLM if args.model_type == "t5" else AutoModelForSequenceClassification
    )
    model = model_class.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    names = ("id", "query", "positive", "negative")
    dataset = (
        load_dataset(
            "csv",
            data_files=args.triples,
            sep="\t",
            split="train",
            names=names,
            features=Features(dict(zip(names, (Value("string"),) * len(names)))),
        )
        .map(infer_type)
        .shuffle()
    )

    if len(dataset) > total_examples:
        unique_ids = dataset.unique("id")
        n_unique_ids = len(unique_ids)
        if n_unique_ids <= total_examples:
            per_id_freq = total_examples // n_unique_ids
            df = dataset.to_pandas()
            df = df.groupby("id").head(per_id_freq)
            dataset = Dataset.from_pandas(df)
        else:
            splits = []
            acronyms = dataset.filter(lambda x: x["type"] == "acronym").to_pandas()
            acronyms = acronyms.groupby("id").first()
            splits.append(Dataset.from_pandas(acronyms))

            n_remaining = total_examples - len(acronyms)
            if n_remaining < 0:
                raise ValueError("Not enough examples to keep all acronyms.")

            keywords = dataset.filter(lambda x: x["type"] == "keyword").to_pandas()
            inpars = dataset.filter(lambda x: x["type"] == "inpars").to_pandas()
            n_keyword_ids = len(keywords["id"].unique())
            n_inpars_ids = len(inpars["id"].unique())
            keyword_ratio = n_keyword_ids / (n_keyword_ids + n_inpars_ids)
            n_keyword_keep = int(n_remaining * keyword_ratio)
            n_inpars_keep = n_remaining - n_keyword_keep

            keywords = keywords.groupby("id").first().head(n_keyword_keep)
            inpars = inpars.groupby("id").first().head(n_inpars_keep)
            splits.append(Dataset.from_pandas(keywords))
            splits.append(Dataset.from_pandas(inpars))

            dataset = concatenate_datasets(splits).shuffle()

    assert len(dataset) == total_examples, f"{len(dataset)} != {total_examples}"

    # with open("queries2.txt", "w") as f:
    #     import csv
    #     w = csv.writer(f, delimiter="\t", lineterminator="\n")
    #     for triple in dataset.sort("id"):
    #         w.writerow([triple["id"], triple["query"]])

    remove_cols = ("id", "type", "query", "positive", "negative")
    if args.model_type == "bert":
        dataset = dataset.map(
            split_triples_bert,
            remove_columns=remove_cols,
            batched=True,
            desc="Splitting triples",
        ).map(
            tokenize_bert,
            remove_columns=("queries", "docs"),
            batched=True,
            desc="Tokenizing",
        )
    elif args.model_type == "t5":
        dataset = dataset.map(
            split_triples_t5,
            remove_columns=remove_cols,
            batched=True,
            desc="Splitting triples",
        ).map(
            tokenize_t5,
            remove_columns=("text", "label"),
            batched=True,
            desc="Tokenizing",
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    trainer_class = (
        Trainer if args.model_type == "bert" else Seq2SeqTrainer
    )

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "args": training_args,
        "train_dataset": dataset,
    }
    if args.model_type == "t5":
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer)

    trainer = trainer_class(**trainer_kwargs)
    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_metrics.metrics)
