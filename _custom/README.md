## Training with InPars for a non-BEIR dataset
We assume that the following files are available:
- **corpus.jsonl**
  - `_id` (str): a unique identifier for the document)
  - `text` (str): the document's content
  - `title` (str): the document's title - can be blank
  - `metadata` (any): any additional metadata for the document - can be blank
- **queries.jsonl**
  - `query_id` (str): a unique identifier for the generated query
  - `query` (str): the generated query text
  - `doc_id` (str): the identifier of the document it was generated from
  - `doc_text` (str): the text of the document it was generated from
- **hard_negatives.tsv**
  - Format: `query_id \t pos_doc_id \t neg_doc_id1 \t neg_doc_id2 ...`
  - `query_id` and `doc_id` correspond to entries from the previous files

### Optional: Filtering
We can filter the generated queries to maintain a high quality of training data. This is recommended if #queries > 10K. To do so, set the variables `CORPUS_FILE` and `QUERIES_FILE` in [filter.sh](./filter.sh) and execute it from the root directory of this repository.

If you want to exclude specific queries from being filtered, you can create a new queries file to pass into the filtering script, and after the filtering is done, merge the filtered queries with the excluded queries with a command like `cat queries_filtered.jsonl queries_excluded.jsonl > queries_final.jsonl`.

**NB:** If you do perform query filtering, make sure to use the new queries file (auto-generated as `queries_filtered.jsonl` in the same directory as the original queries) in the following steps.

### Fine-tuning a bi-encoder
To fine-tune a bi-encoder (default base model is [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)), set the variables `CORPUS_FILE`, `QUERIES_FILE` and `NEGS_FILE` in [train_be.sh](./train_be.sh) and execute it from the root directory of this repository. The trained model will be saved in a new directory named `models/mpnet_sbert` in the same directory as the provided queries.

Behind the scenes, this extracts a qrels.tsv file from the provided queries, marking the original document as relevant, and tokenizes the corpus and queries, storing them as a HuggingFace dataset to be used with the training script.

### Fine-tuning a cross-encoder
To fine-tune a cross-encoder (default base model is [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)), set the variables `CORPUS_FILE`, `QUERIES_FILE` and `NEGS_FILE` in [train_ce.sh](./train_ce.sh) and execute it from the root directory of this repository. The trained model will be saved in a new directory named `models/miniLM` in the same directory as the provided queries.

Behind the scenes, this converts the hard negatives to a triples.tsv file which contains a (query, positive document, negative document) pair in each line, to be used with the training script.
