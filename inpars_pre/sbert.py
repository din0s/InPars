"""Saves a trained model in the SentenceTransformer format."""
import argparse
from collections import OrderedDict

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    transformer = Transformer(args.model_path)
    pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
    modules = [transformer, pooling]

    modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
    model = SentenceTransformer(modules=modules)
    model.save(args.output_path)
