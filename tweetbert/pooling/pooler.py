from typing import (
    List,
    # Optional
)
import numpy as np

# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer


# class Pooler(FunctionTransformer):
#     def __init__(self, pooling_method: Optional[str] = None):
#         self.pooling_method = pooling_method

#     def fit(self, X: List[List[np.array]], y=None):
#         return self

#     def transform(self, X: List[List[np.array]]):
#         return np.stack([pool_subwords(i, self.pooling_method) for i in X])

#     def fit_transform(self, X: List[List[np.array]], y=None):
#         return np.stack([pool_subwords(i, self.pooling_method) for i in X])


def pool_subwords(
    subword_representations: List[np.ndarray], pooling_method: str
) -> np.ndarray:
    """
    Given a list of at least token representations for a word,
    applies pooling
    Works on words that have exactly 1 or more than 1 subword representation
    """
    if pooling_method == "first":
        # returns array of dimension (hidden_size, )
        return subword_representations[0]

    elif pooling_method == "concat_first_last":
        # returns array of dimension (2*hidden_size, )
        return np.concatenate(
            [subword_representations[0], subword_representations[-1]], axis=0
        )
    elif pooling_method == "mean":
        # returns array of dimension (hidden_size, )
        return np.mean(subword_representations, axis=0)

    elif pooling_method == "max":
        # elementwise max and returns array of dimension (hidden_size)
        return np.mean(subword_representations, axis=0)

    elif pooling_method == "min":
        # elementwise min and returns array of dimension (hidden_size)
        return np.min(subword_representations, axis=0)

    elif pooling_method == "std":
        # elementwise std and returns array of dimension (hidden_size)
        return np.std(subword_representations, axis=0)

    else:
        return None
