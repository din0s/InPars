import json
import os
import random

import ftfy
import yaml

with open(f"{os.path.dirname(__file__)}/prompts/templates.yaml") as f:
    templates = yaml.safe_load(f)


class Prompt:
    def __init__(
        self,
        index=None,
        instruction=None,
        template=None,
        examples=None,
        tokenizer=None,
        max_doc_length=None,
        max_query_length=None,
        max_prompt_length=None,
        max_new_token=16,
        **kwargs,
    ):
        self.index = index
        self.instruction = instruction
        self.template = template
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_token = max_new_token

    @classmethod
    def load(cls, name, *args, **kwargs):
        if name in templates:
            template = templates[name]
            prompt_class = {
                "dynamic": DynamicPrompt,
                "static": StaticPrompt,
                "alpaca": AlpacaPrompt,
                "instruction": InstructionPrompt,
                "causal-instruction": CausalInstructionPrompt,
                "co-instruction": ContrastiveInstructionPrompt,
                "chat": ChatPrompt,
                "co-chat": ContrastiveChatPrompt,
            }[template["mode"]]
            return prompt_class(*args, **template, **kwargs)
        else:
            if not os.path.exists(name):
                raise FileNotFoundError(f"Prompt file {name} was not found!")

            with open(name) as f:
                return StaticPrompt(template=f.read(), *args, **kwargs)

    def _truncate_max_doc_length(self, document):
        if self.max_doc_length:
            document = self.tokenizer.decode(
                self.tokenizer(
                    document,
                    truncation=True,
                    max_length=self.max_doc_length,
                )["input_ids"]
            )
        return document


class StaticPrompt(Prompt):
    def build(self, document, *args, **kwargs):
        document = self._truncate_max_doc_length(document)

        prompt = self.template.format(document=document, query="").rstrip()

        if self.max_prompt_length:
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )

        return prompt


class DynamicPrompt(Prompt):
    def build(self, text, n_examples=3):
        random_examples = random.sample(self.examples, n_examples)

        prompt = self._get_base_prompt()
        for i in range(n_examples):
            _, _, query, doc = random_examples[i]
            query = self._truncate_max_query_length(query)
            doc = self._truncate_max_doc_length(doc)

            prompt += self._format_template(doc, query, i)

        if self._append_suffix():
            prompt += text.rstrip()
        else:
            document = ftfy.fix_text(text)
            if self.max_doc_length:
                document = self.tokenizer.decode(
                    self.tokenizer(
                        document,
                        truncation=True,
                        max_length=self.max_doc_length,
                    )["input_ids"]
                )

            prompt += self._format_template(document, "", n_examples).rstrip()

        if self.max_prompt_length:
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )

        return prompt

    def _get_base_prompt(self):
        return ""

    def _append_suffix(self):
        return False

    def _format_template(self, document, query, example_idx):
        args = {
            "document": document,
            "query": query,
        }
        if "example_idx" in self.template:
            args["example_idx"] = example_idx + 1

        return self.template.format(**args)

    def _truncate_max_query_length(self, query):
        if self.max_query_length:
            query = self.tokenizer.decode(
                self.tokenizer(
                    query,
                    truncation=True,
                    max_length=self.max_query_length,
                )["input_ids"]
            )
        return query


class AlpacaPrompt(DynamicPrompt):
    def _get_base_prompt(self):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{self.instruction}\n"
        )


class InstructionPrompt(DynamicPrompt):
    def _get_base_prompt(self):
        return self.instruction + " Only respond with the generated query.\n\n"


class CausalInstructionPrompt(DynamicPrompt):
    def _get_base_prompt(self):
        return self.instruction + "\n\n"

    def _append_suffix(self):
        return True


class ContrastiveInstructionPrompt(InstructionPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.topk = 50

        from pyserini.search.lucene import LuceneSearcher

        if os.path.isdir(self.index):
            self.searcher = LuceneSearcher(self.index)
        else:
            self.searcher = LuceneSearcher.from_prebuilt_index(self.index)

    def _get_base_prompt(self):
        return (
            self.instruction
            + " However, make sure that the generated query does not follow the "
            + 'instruction for the "Irrelevant Document". Only respond with the '
            + "generated query.\n\n"
        )

    def _format_template(self, rel_document, query, example_idx):
        results = self.searcher.search(rel_document, k=self.topk + 1)
        hit = json.loads(results[self.topk].raw)
        irr_document = ftfy.fix_text(
            f"{hit['title']} {hit['text']}" if hit["title"] else hit["text"]
        )

        args = {
            "rel_document": rel_document,
            "irr_document": irr_document,
            "query": query,
        }
        if "example_idx" in self.template:
            args["example_idx"] = example_idx + 1

        return self.template.format(**args)


class ChatPrompt(DynamicPrompt):
    def build(self, document, n_examples=3):
        random_examples = random.sample(self.examples, n_examples)

        messages = [{"role": "system", "content": self._get_instruction()}]

        for i in range(n_examples):
            _, _, query, doc = random_examples[i]
            query = self._truncate_max_query_length(query)
            doc = self._truncate_max_doc_length(doc)

            messages += [
                {"role": "user", "content": self._format_template(doc)},
                {"role": "assistant", "content": query},
            ]

        document = ftfy.fix_text(document).rstrip()
        if self.max_doc_length:
            document = self.tokenizer.decode(
                self.tokenizer(
                    document, truncation=True, max_length=self.max_doc_length
                )["input_ids"]
            )

        messages += [
            {"role": "user", "content": self._format_template(document)},
        ]

        if self.max_prompt_length:
            prompt = " ".join([m["content"] for m in messages])
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )

        return messages

    def _get_instruction(self):
        return self.instruction + " Only respond with the generated query."

    def _format_template(self, document):
        return self.template.format(document=document).rstrip()


class ContrastiveChatPrompt(ChatPrompt):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.topk = 50

        from pyserini.search.lucene import LuceneSearcher

        if os.path.isdir(self.index):
            self.searcher = LuceneSearcher(self.index)
        else:
            self.searcher = LuceneSearcher.from_prebuilt_index(self.index)

    def _get_instruction(self):
        return (
            self.instruction
            + " However, make sure that the generated query does not follow the "
            + 'instruction for the "Irrelevant Document". Only respond with the '
            + "generated query."
        )

    def _format_template(self, rel_document):
        results = self.searcher.search(rel_document, k=self.topk + 1)
        hit = json.loads(results[self.topk].raw)
        irr_document = ftfy.fix_text(
            f"{hit['title']} {hit['text']}" if hit["title"] else hit["text"]
        )
        return self.template.format(
            rel_document=rel_document,
            irr_document=irr_document,
        ).rstrip()
