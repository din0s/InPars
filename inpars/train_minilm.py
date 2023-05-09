from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
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
    max_doc_length: int = field(
        default=300,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )


def split_triples(triples):
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


def tokenize(batch):
    tokenized = tokenizer(
        batch["queries"],
        batch["docs"],
        padding=True,
        truncation="longest_first",
        max_length=args.max_doc_length,
    )
    return tokenized


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ExtraArguments))
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
            * torch.cuda.device_count()
        )

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = load_dataset(
        "csv",
        data_files=args.triples,
        sep="\t",
        names=("query", "positive", "negative"),
    )
    dataset = dataset.map(
        split_triples,
        remove_columns=("query", "positive", "negative"),
        batched=True,
    )
    if total_examples:
        dataset["train"] = dataset["train"].shuffle().select(range(total_examples))
    dataset = dataset.map(
        tokenize,
        remove_columns=("queries", "docs"),
        batched=True,
        desc="Tokenizing",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
    )
    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_metrics.metrics)
