from torch.utils.data import Dataset
from typing import List, Optional
from sqlalchemy.engine.base import Engine

# These data structures are designed for language model training


class TweetDataset(Dataset):
    """
    Dataset of tweets with optional doc-level labels.
    Make a TweetDataset out of the tweets from a LmCorpus
    """

    def __init__(self, tweets: List[str]):
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index: int):
        return self.tweets[index]


class LmCorpus:
    """
    For language model training
    """

    def __init__(
        self,
        db_engine: Engine,
        db_table: str,
        db_col: str,
        order_col: str = "id",
    ):
        self.db_engine = db_engine
        self.db_table = db_table
        self.db_col = db_col

        result = db_engine.execute(
            f"SELECT {db_col} FROM {db_table} ORDER BY {order_col}"
        )  # each row is a tuple
        self.tweets = [r[0] for r in result]
        self.size = len(self.tweets)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.tweets[index]
