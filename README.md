<h1 align="center">InPars</h1>
<div align="center">
  <strong>Inquisitive Parrots for Search</strong>
</div>
<div align="center">
  A toolkit for end-to-end synthetic data generation using LLMs for IR
</div>
<div align="center">
	<a href="https://pypi.org/project/inpars/">
		<img src="https://img.shields.io/pypi/v/inpars?style=flat-square">
	</a>
	<a href="">
		<img src="https://img.shields.io/github/contributors/zetaalphavector/inpars?style=flat-square">
	</a>
</div>

<div align="center">
  <h3>
    <a href="#Installation">Installation</a>
    <span> | </span>
    <a href="#Usage">Usage</a>
    <span> | </span>
    <a href="#Resources">Resources</a>
    <span> | </span>
    <a href="#Contributing">Contributing</a>
    <span> | </span>
    <a href="#References">References</a>
  </h3>
</div>


## Installation

Use pip package manager to install InPars toolkit.

```bash
pip install inpars
```

## Usage

To generate data for one of the [BEIR datasets](https://github.com/beir-cellar/beir), you can use the following command:

```bash
python -m inpars.generate \
        --prompt="inpars" \
        --dataset="trec-covid" \
        --dataset_source="ir_datasets" \
        --base_model="EleutherAI/gpt-j-6B" \
        --output="trec-covid-queries.jsonl" 
```

Additionally, you can use your own custom dataset by specifying the `corpus` and `queries` arguments to local files.

These generated queries might be noisy, thus a filtering step is highly recommended:

```bash
python -m inpars.filter \
        --input="trec-covid-queries.jsonl" \
        --dataset="trec-covid" \
        --filter_strategy="scores" \
        --keep_top_k="10_000" \
        --output="trec-covid-queries-filtered.jsonl"
```

There are currently two filtering strategies available: scores, which uses probability scores from the LLM itself, and reranker, which uses an auxiliary reranker to filter queries as introduced by [InPars-v2](https://arxiv.org/abs/2301.01820).

To prepare the training file, negative examples are mined by retrieving candidate documents with BM25 using the generated queries and sampling from these candidates. This is done using the following command:

```bash
python -m inpars.generate_triples \
        --input="trec-covid-queries-filtered.jsonl" \
        --dataset="trec-covid" \
        --output="trec-covid-triples.tsv"
```

With the generated triples file, you can train the model using the following command:

```bash
python -m inpars.train \
        --triples="trec-covid-triples.tsv" \
        --base_model="castorini/monot5-3b-msmarco-10k" \
        --output_dir="./reranker/" \
        --max_steps="156"
```

You can choose different base models, hyperparameters, and training strategies that are supported by [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).

After finetuning the reranker, you can rerank prebuilt runs from the BEIR benchmark or specify a custom run using the following command:

```bash
python -m inpars.rerank \
        --model="./reranker/" \
        --dataset="trec-covid" \
        --output_run="trec-covid-run.txt"
```

Finally, you can evaluate the reranked run using the following command:

```bash
python -m inpars.evaluate \
        --dataset="trec-covid" \
        --run="trec-covid-run.txt"
```

## Resources

#### Generated datasets

Download synthetic datasets generated by InPars-v1:

- [MS-MARCO / TREC-DL](https://zav-public.s3.amazonaws.com/inpars/msmarco_synthetic_queries_100k.jsonl)
- [Robust04](https://zav-public.s3.amazonaws.com/inpars/robust04_synthetic_queries_100k.jsonl)
- [Natural Questions](https://zav-public.s3.amazonaws.com/inpars/nq_synthetic_queries_100k.jsonl)
- [TREC-COVID](https://zav-public.s3.amazonaws.com/inpars/trec_covid_synthetic_queries_100k.jsonl)
- [FiQA](https://zav-public.s3.amazonaws.com/inpars/fiqa_synthetic_queries_100k.jsonl)
- [DBPedia](https://zav-public.s3.amazonaws.com/inpars/dbpedia_synthetic_queries_100k.jsonl)
- [SCIDOCS](https://zav-public.s3.amazonaws.com/inpars/scidocs_synthetic_queries_100k.jsonl)
- [SciFact](https://zav-public.s3.amazonaws.com/inpars/scifacts_synthetic_queries_100k.jsonl)
- [ArguAna](https://zav-public.s3.amazonaws.com/inpars/arguana_synthetic_queries_100k.jsonl)
- [BioASQ](https://zav-public.s3.amazonaws.com/inpars/bioasq_synthetic_queries_100k.jsonl)
- [Climate Fever](https://zav-public.s3.amazonaws.com/inpars/climate_fever_synthetic_queries_100k.jsonl)
- [CQADupstack](https://zav-public.s3.amazonaws.com/inpars/cqadupstack_synthetic_queries_100k.jsonl)
- [Fever](https://zav-public.s3.amazonaws.com/inpars/fever_synthetic_queries_100k.jsonl)
- [Hotpotqa](https://zav-public.s3.amazonaws.com/inpars/hotpotqa_synthetic_queries_100k.jsonl)
- [NFCorpus](https://zav-public.s3.amazonaws.com/inpars/nfcorpus_synthetic_queries_100k.jsonl)
- [Quora](https://zav-public.s3.amazonaws.com/inpars/quora_synthetic_queries_100k.jsonl)
- [Signal-1M](https://zav-public.s3.amazonaws.com/inpars/signal_synthetic_queries_100k.jsonl)
- [Touche-2020](https://zav-public.s3.amazonaws.com/inpars/touche_synthetic_queries_100k.jsonl)
- [TREC-NEWS](https://zav-public.s3.amazonaws.com/inpars/trec_news_synthetic_queries_100k.jsonl)

#### Finetuned models

Download finetuned models from InPars-v2 on [HuggingFace Hub](https://huggingface.co/models?search=zeta-alpha-ai/monot5-3b-inpars-v2-).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.


## References

Currently, if you use this tool you can cite the original [InPars paper published at SIGIR](https://dl.acm.org/doi/10.1145/3477495.3531863) or [InPars-v2](https://arxiv.org/abs/2301.01820).

```
@inproceedings{inpars,
  author = {Bonifacio, Luiz and Abonizio, Hugo and Fadaee, Marzieh and Nogueira, Rodrigo},
  title = {{InPars}: Unsupervised Dataset Generation for Information Retrieval},
  year = {2022},
  isbn = {9781450387323},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3477495.3531863},
  doi = {10.1145/3477495.3531863},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages = {2387–2392},
  numpages = {6},
  keywords = {generative models, large language models, question generation, synthetic datasets, few-shot models, multi-stage ranking},
  location = {Madrid, Spain},
  series = {SIGIR '22}
}
```

```
@misc{inparsv2,
  doi = {10.48550/ARXIV.2301.01820},
  url = {https://arxiv.org/abs/2301.01820},
  author = {Jeronymo, Vitor and Bonifacio, Luiz and Abonizio, Hugo and Fadaee, Marzieh and Lotufo, Roberto and Zavrel, Jakub and Nogueira, Rodrigo},
  title = {{InPars-v2}: Large Language Models as Efficient Dataset Generators for Information Retrieval},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```