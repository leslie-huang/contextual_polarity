from typing import List, Optional
import pandas as pd
from torch.utils.data import Dataset
from tweetbert.corpus.base_corpus import BaseCorpus
from tweetbert.corpus.utils import spacify_texts


def filter_df_for_word(word: str, contexts_df: pd.DataFrame, nlp, matcher):
    """
    Recheck for the target word in a set of tokenized contexts,
    and returns a dataframe that includes a column for spacy
    tokenized contexts
    """
    all_tokenized_contexts = spacify_texts(
        contexts_df["text"].tolist(), nlp, matcher
    )
    mask = []
    # now check that the word is actually there in the tokenized contexts
    # everything has already been lowercased if we want case insensitive
    for tokenized_context in all_tokenized_contexts:
        if word not in tokenized_context:
            mask.append(False)
        else:
            mask.append(True)

    # only select contexts where the word was found
    filtered_contexts_df = contexts_df[mask]
    return filtered_contexts_df


class TargetWordCorpus:
    """
    This data structure holds all (up to the specified limit) tweets
    containing a target word. (Limit is necessary because some words
    occur 1+ million times)
    We check AGAIN with spacy tokenization for the word
    These are RANDOMLY selected, stratified by class
    contexts_by_class is a Dict, {class_name: dataframe of tweets}
    """

    def __init__(self, corpus: BaseCorpus, word: str, limit: int = 10000):
        self.limit = limit  # max number of data points to return PER CLASS
        self.word = word

        self.corpus_metadata = {
            "n_classes": corpus.n_classes,
            "class_names": corpus.class_names,
            "class_var": corpus.class_var,
        }

        if corpus.word_counts:
            vocab_list = list(corpus.word_counts.keys())
        elif corpus.word_doc_counts:
            vocab_list = list(corpus.word_doc_counts.keys())

        if self.word not in vocab_list:
            print(f"Warning: {word} is not in the vocab of this corpus!")

        self.contexts_by_class = {}

        for class_level in corpus.class_names:
            contexts_df = pd.read_sql_query(
                f"SELECT * FROM {corpus.db_table} \
                        WHERE {corpus.class_var} = {class_level} AND \
                        {corpus.text_col} LIKE '% {self.word} %' OR \
                        {corpus.text_col} LIKE '% {self.word}' OR \
                        {corpus.text_col} LIKE '{self.word} %' OR \
                        {corpus.text_col} LIKE '{self.word}' \
                        ORDER BY RANDOM() LIMIT {self.limit} ",
                corpus.db_engine,
            )
            contexts_df.rename(columns={corpus.text_col: "text"}, inplace=True)

            filtered_contexts_df = filter_df_for_word(
                self.word, contexts_df, corpus.nlp, corpus.matcher
            )

            self.contexts_by_class[class_level] = filtered_contexts_df

        self.class_counts = {
            k: v.shape[0] for k, v in self.contexts_by_class.items()
        }

    def __len__(self):
        return sum(self.class_counts.values())

    def get_sample(self, n_requested: int, sample_min_per_class: float = 0.4):
        self.n_requested = n_requested
        self.sample_min_per_class = sample_min_per_class
        self.too_imbalanced = None

        for class_level, n_samples in self.class_counts.items():
            if n_samples < sample_min_per_class:
                print(
                    f"Warning: {n_samples} not {sample_min_per_class} docs for class {class_level}"
                )
                self.too_imbalanced = True
            else:
                self.too_imbalanced = False

        self.selected_df = self._select_tweets()

    def _select_tweets(self):
        """
        Returns df of tweets across all class levels
        """
        records = []
        dfs_as_dicts = [
            i.to_dict(orient="records")
            for i in self.contexts_by_class.values()
        ]
        pop_size = sum([len(j) for j in dfs_as_dicts])
        dicts_iterators = [iter(i) for i in dfs_as_dicts]

        while len(records) < min(self.n_requested, pop_size):
            for i in dicts_iterators:
                res = next(i, None)
                if res is not None:
                    records.append(res)

        selected_df = pd.DataFrame.from_dict(records)

        return selected_df


class TargetWordDataset(Dataset):
    """
    A Dataset containing a TargetWordCorpus sample
    """

    def __init__(
        self,
        twc: TargetWordCorpus,
        metadata_fields: List = ["id"],
        tokenized_max_length: Optional[int] = None,
    ):
        self.tweets = twc.selected_df["text"].tolist()

        self.labels = twc.selected_df[
            twc.corpus_metadata["class_var"]
        ].tolist()

        if len(metadata_fields) > 0:
            self.metadata = twc.selected_df[metadata_fields].to_dict(
                orient="records"
            )

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index: int):
        return (self.tweets[index], self.labels[index])
