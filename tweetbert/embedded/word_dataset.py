from typing import List, Dict, Optional
import numpy as np
from tweetbert.pooling.pooler import pool_subwords
from collections import OrderedDict, Counter
import logging


class WordDataset:
    def __init__(self, word: str, pooling_method: Optional[str] = None):
        self.data = OrderedDict()
        self.word = word
        self.pooling_method = pooling_method
        self.class_distr = None
        self.labels = None
        self.is_imbalanced = False

    def add_document(
        self,
        doc_id: str,
        label: str,
        metadata: Dict,
        doc_representation: np.array,
        reconstructed_tokens: Optional[List] = None,
    ):
        """
        self.data is a Dict of k=label, v=Dict
        that Dict has k=doc_id, v=Dict of doc data
        Add a document, and pooling will automatically be
        applied to the representation
        """
        if reconstructed_tokens is not None:
            if self.word in reconstructed_tokens:
                word_idx = reconstructed_tokens.index(self.word)
                rep = doc_representation[word_idx]
                if self.pooling_method:
                    rep = pool_subwords(rep, self.pooling_method)

                if label not in self.data.keys():
                    self.data[label] = OrderedDict()

                self.data[label][doc_id] = {
                    "id": doc_id,
                    "label": label,
                    "representation": rep,
                    "metadata": metadata,
                    "reconstructed_tokens": reconstructed_tokens,
                }
        else:
            if label not in self.data.keys():
                self.data[label] = OrderedDict()

            # it's a fasttext doc and we've already gotten the representation for the token
            self.data[label][doc_id] = {
                "id": doc_id,
                "label": label,
                "representation": doc_representation,  # this is actually sentence minus target
                "metadata": metadata,
                "reconstructed_tokens": reconstructed_tokens,  # which is None
            }

    def __len__(self):
        if self.labels is None:
            logging.info("Need to get_sample() first")
            return None
        else:
            return len(self.labels)

    def _get_class_distr(self):
        self.class_distr = {
            label: len(self.data[label]) for label in self.data.keys()
        }
        if len(self.class_distr) == 1:
            self.is_imbalanced = True

    def get_sample(
        self, n_total: Optional[int] = None, min_per_class: int = 0
    ):
        """
        Gets a sample of representations, labels, and metadata
        This can be done repeatedly
        """
        # first get class breakdown
        self._get_class_distr()
        if n_total is None:
            n_total = sum(self.class_distr.values())

        if self.is_imbalanced is True:
            return None

        for label, num_docs in self.class_distr.items():
            if num_docs < min_per_class:
                logging.info(
                    f"Stopping word {self.word}: {num_docs} for label {label}"
                )
                self.is_imbalanced = True
                return None

        doc_iterators = [iter(list(d.items())) for k, d in self.data.items()]
        combined_data = []
        while len(combined_data) < min(
            n_total, sum(self.class_distr.values())
        ):
            for i in doc_iterators:
                res = next(i, None)
                if res is not None:
                    combined_data.append(res[1])

        # now we have to convert this into lists and stack the numpy arrays
        if combined_data and not self.is_imbalanced:
            self.labels = [i["label"] for i in combined_data]
            self.metadata = [i["metadata"] for i in combined_data]
            self.pooled_X = np.stack(
                [i["representation"] for i in combined_data]
            )
            self.sample_distr = Counter(self.labels)

    def reduce_size(self, n: int, min_per_class: int = 0):
        self.labels = self.labels[:n]
        self.metadata = self.metadata[:n]
        self.pooled_X = self.pooled_X[:n]
        self.sample_distr = Counter(self.labels)

        for label, num_docs in self.sample_distr.items():
            if num_docs < min_per_class:
                logging.info(
                    f"Stopping word {self.word}: {num_docs} for label {label}"
                )
                self.is_imbalanced = True
