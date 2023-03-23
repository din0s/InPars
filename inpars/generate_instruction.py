import argparse

from transformers import set_seed

from .inpars import InPars

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--dataset", required=True, help="Dataset name from BEIR")
    parser.add_argument("--n_fewshot_examples", type=int, default=3)
    parser.add_argument("--max_doc_length", default=256, type=int, required=False)
    parser.add_argument("--max_query_length", default=200, type=int, required=False)
    parser.add_argument("--max_prompt_length", default=2048, type=int, required=False)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_batch_size", type=int, default=1)
    # parser.add_argument('--max_generations', type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--tf", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--is_openai", action="store_true")
    parser.add_argument("--is_llama", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)

    generator = InPars(
        base_model=args.base_model,
        revision=args.revision,
        corpus=args.dataset,
        prompt="instruction-extract",
        n_fewshot_examples=args.n_fewshot_examples,
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        max_batch_size=args.max_batch_size,
        fp16=args.fp16,
        int8=args.int8,
        tf=args.tf,
        device=args.device,
        is_openai=args.is_openai,
        is_llama=args.is_llama,
        # verbose=args.verbose,
    )

    generate_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.is_llama:
        generate_kwargs["top_k"] = args.top_k
        generate_kwargs["repetition_penalty"] = args.repetition_penalty
    elif not args.is_openai:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    instruction = generator.generate_instruction(**generate_kwargs)
    with open(args.output, "w") as f:
        f.write(instruction)
