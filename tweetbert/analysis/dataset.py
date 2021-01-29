import pandas as pd
from torch.utils.data import Dataset
from typing import List, Optional


class SimpleDataset(Dataset):
    """
    A Dataset containing a sample of tweets for a target word
    """

    def __init__(
        self,
        df: pd.DataFrame,
        word: str,
        label_var: str = "label",
        metadata_fields: List = ["id"],
        tokenized_max_length: Optional[int] = None,
    ):
        self.tweets = df["text"].tolist()
        self.word = word

        self.labels = df[label_var].tolist()

        if len(metadata_fields) > 0:
            self.metadata = df[metadata_fields].to_dict(orient="records")

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index: int):
        return (self.tweets[index], self.labels[index])
