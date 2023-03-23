from .model import FewShotModel


class InPars:
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
        fp16=False,
        int8=False,
        device=None,
        tf=False,
        verbose=False,
        is_openai=False,
    ):
        self.model = FewShotModel(
            base_model=base_model,
            lora_weights=lora_weights,
            revision=revision,
            corpus=corpus,
            prompt=prompt,
            n_fewshot_examples=n_fewshot_examples,
            max_doc_length=max_doc_length,
            max_query_length=max_query_length,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            fp16=fp16,
            int8=int8,
            device=device,
            tf=tf,
            verbose=verbose,
            is_openai=is_openai,
        )

    def generate_instruction(self, **generate_kwargs):
        instructions = list(self.model.generate(["Instruction:"], **generate_kwargs))
        return instructions[0][0]["output"]

    def generate_queries(
        self,
        documents,
        doc_ids,
        batch_size=1,
        **generate_kwargs,
    ):
        fewshot_examples = [example[0] for example in self.model.fewshot_examples]
        for batch_idx, batch_results in enumerate(
            self.model.generate(
                inputs=documents,
                batch_size=batch_size,
                **generate_kwargs,
            )
        ):
            batch_doc_ids = doc_ids.iloc[
                batch_idx * batch_size : (batch_idx + 1) * batch_size
            ]
            for idx, result in enumerate(batch_results):
                yield {
                    "query": result["output"],
                    "log_probs": result["probs"],
                    "prompt_text": result["prompt"],
                    "doc_id": batch_doc_ids.iloc[idx],
                    "doc_text": result["input"],
                    "fewshot_examples": fewshot_examples,
                }

