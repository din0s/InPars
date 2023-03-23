import os
import random
from functools import partial

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import Prompt


class FewShotModel:
    def __init__(
        self,
        base_model="EleutherAI/gpt-j-6B",
        lora_weights=None,
        revision=None,
        corpus=None,
        prompt=None,
        n_fewshot_examples=None,
        max_doc_length=None,
        max_query_length=None,
        max_prompt_length=None,
        max_new_tokens=64,
        max_batch_size=1,
        fp16=False,
        int8=False,
        device=None,
        tf=False,
        verbose=False,
        is_openai=False,
    ):
        self.corpus = corpus
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = max_batch_size
        self.n_fewshot_examples = n_fewshot_examples
        self.device = device
        self.tf = tf
        self.verbose = verbose
        self.is_openai = is_openai
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.is_openai:
            self.openai_engine = base_model
            base_model = "gpt2"

        if "llama" in base_model:
            from transformers import LlamaTokenizer

            self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model, padding_side="left"
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Hardcode \n token for EOS
        self.newline_token_id = None
        if base_model == "EleutherAI/gpt-j-6B":
            self.newline_token_id = 198
        elif "llama" in base_model:
            self.newline_token_id = 13

        model_kwargs = {"revision": revision}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        if int8:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"

        if self.is_openai:
            import openai

            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif self.tf:
            from transformers import TFAutoModelForCausalLM

            self.model = TFAutoModelForCausalLM.from_pretrained(
                base_model,
                revision=revision,
            )
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, **model_kwargs
            )
            if lora_weights is not None:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_weights,
                    torch_dtype=model_kwargs.get("torch_dtype", None),
                )
            self.model = torch.compile(self.model)
            self.model.to(self.device)
            self.model.eval()

        self.fewshot_examples = self._load_examples()
        self.prompter = Prompt.load(
            name=prompt,
            examples=self.fewshot_examples,
            tokenizer=self.tokenizer,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_prompt_length=self.max_prompt_length,
            max_new_token=self.max_new_tokens,
        )

    def _load_examples(self):
        if os.path.exists(self.corpus):
            df = pd.read_json(self.corpus, lines=True)
        else:
            df = pd.read_json(
                f"https://huggingface.co/datasets/inpars/fewshot-examples/resolve/main/data/{self.corpus}.json",
                lines=True,
            )
        # TODO limitar numero de exemplos (?)
        df = df[["query_id", "doc_id", "query", "document"]].values.tolist()
        random_examples = random.sample(df, self.n_fewshot_examples)
        with open("query_ids_to_remove_from_eval.tsv", "w") as fout:
            for item in random_examples:
                fout.write(f"{item[0]}\t{item[2]}\n")

        return random_examples

    def _build_prompts(self, inputs):
        disable_pbar = False if len(inputs) > 1_000 else True
        prompts = [
            self.prompter.build(text, n_examples=self.n_fewshot_examples)
            for text in tqdm(inputs, disable=disable_pbar, desc="Building prompts")
        ]

        return prompts

    @torch.no_grad()
    def generate(self, inputs, batch_size=1, **kwargs):
        if self.is_openai:
            if "gpt-3.5" in self.openai_engine:
                return self._generate_openai_chat(inputs, batch_size, **kwargs)
            else:
                return self._generate_openai_complete(inputs, batch_size, **kwargs)
        else:
            return self._generate_hf(inputs, batch_size, **kwargs)

    def _generate_openai_chat(self, inputs, batch_size=1, **kwargs):
        assert batch_size == 1, "Batch size must be 1 for OpenAI Chat API"

        import openai

        generate = partial(
            openai.ChatCompletion.create,
            model=self.openai_engine,
            max_tokens=self.max_new_tokens,
            **kwargs,
        )

        prompts = self._build_prompts(inputs)
        for i, prompt in tqdm(enumerate(prompts), desc="Generating text"):
            prompt_text = " ".join(
                [f'[{message["role"]}] {message["content"]}' for message in prompt]
            )
            api_response = generate(messages=prompt)
            output = api_response["choices"][0]["message"]["content"]
            yield Result([inputs[i]], [prompt_text], [output])

    def _generate_openai_complete(self, inputs, batch_size=1, **kwargs):
        import openai

        generate = partial(
            openai.Completion.create,
            model=self.openai_engine,
            max_tokens=self.max_new_tokens,
            echo=True,
            **kwargs,
        )

        prompts = self._build_prompts(inputs)
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text"):
            batch_prompts = prompts[i : i + batch_size]
            batch_inputs = inputs[i : i + batch_size]
            api_responses = generate(prompt=batch_prompts)
            batch_outputs = [
                choice["text"].strip() for choice in api_responses["choices"]
            ]
            yield Result(batch_inputs, batch_prompts, batch_outputs)

    def _generate_hf(self, inputs, batch_size=1, **kwargs):
        if self.tf:
            import tensorflow as tf

            generate = tf.function(self.model.generate, jit_compile=True)
            padding_kwargs = {"pad_to_multiple_of": 8}
        else:
            generate = self.model.generate
            padding_kwargs = {}

        prompts = self._build_prompts(inputs)
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text"):
            batch_prompts = prompts[i : i + batch_size]
            batch_inputs = inputs[i : i + batch_size]

            tokens = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="tf" if self.tf else "pt",
                **padding_kwargs,
            )

            if not self.tf:
                tokens.to(self.device)

            batch_preds = generate(
                input_ids=tokens["input_ids"].long(),
                attention_mask=tokens["attention_mask"].long(),
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict=True,
                return_dict_in_generate=True,
                eos_token_id=self.newline_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
            batch_seqs = batch_preds.sequences[:, tokens["input_ids"].shape[-1] :]
            batch_outputs = [
                seq.strip()
                for seq in self.tokenizer.batch_decode(
                    batch_seqs, skip_special_tokens=True
                )
            ]
            batch_scores = torch.stack(batch_preds.scores, dim=1)
            batch_probs, _ = batch_scores.log_softmax(dim=-1).max(dim=2)
            pad_mask = batch_seqs == self.tokenizer.pad_token_id
            batch_probs = [
                batch_probs[i][~pad_mask[i]].tolist()[:-1]
                for i in range(len(batch_probs))
            ]
            yield [
                Result(input, prompt, out, probs)
                for input, prompt, out, probs in zip(
                    batch_inputs, batch_prompts, batch_outputs, batch_probs
                )
            ]


class Result:
    def __init__(self, inputs, prompts, outputs, probs=None):
        assert len(inputs) == len(prompts) == len(outputs)
        self.inputs = inputs
        self.prompts = prompts
        self.outputs = outputs
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [[] for _ in range(len(self.outputs))]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input": self.inputs[idx],
            "prompt": self.prompts[idx],
            "output": self.outputs[idx],
            "probs": self.probs[idx],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
