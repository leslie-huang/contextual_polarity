import logging
import numpy as np
import pandas as pd
import spacy
from spacy import lang
from sqlalchemy.engine.base import Engine
from typing import Dict, List, Optional
from typing_extensions import Literal
from tweetbert.analysis.utils import get_speakers_above_threshold

nlp = spacy.load("en_core_web_sm")


def sample_tweets_containing_word(
    db_engine: Engine,
    word: str,
    db_table: str = "tweets",
    text_col: str = "text",
):
    contexts_df = pd.read_sql_query(
        f"SELECT * FROM {db_table} \
                WHERE {text_col} LIKE '% {word} %' OR \
                {text_col} LIKE '% {word}' OR \
                {text_col} LIKE '{word} %' OR \
                {text_col} LIKE '{word}' ",
        db_engine,
    )
    contexts_df.rename(columns={text_col: "text"}, inplace=True)
    return contexts_df


def get_speakers_subset(
    contexts_df_for_word: pd.DataFrame,
    screen_names: List[str],
    max_per_speaker: Optional[int],
):
    """
    can request one or many speakers
    """
    df = contexts_df_for_word[
        contexts_df_for_word.screen_name.isin(screen_names)
    ]

    if max_per_speaker:
        # only get a certain number of tweets per speaker
        df = (
            df.groupby(["screen_name"])
            .apply(
                lambda grp: grp.sample(n=min(max_per_speaker, grp.shape[0]))
            )
            .reset_index(level=[0, 1], drop=True)
        )

    return df


def get_tweets_above_threshold(
    tweets: pd.DataFrame,
    official_only: bool = True,
    min_threshold: int = 40,
    max_per_speaker: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns {max_per_speaker|all} tweets from speakers
    who tweeted at least min_threshold times
    """
    screen_names_above_cutoff = get_speakers_above_threshold(
        tweets, official_only, min_threshold
    )

    tweets = tweets[tweets.screen_name.isin(screen_names_above_cutoff)]

    if max_per_speaker:
        # only get a certain number of tweets per speaker
        tweets = (
            tweets.groupby(["screen_name"])
            .apply(
                lambda grp: grp.sample(n=min(max_per_speaker, grp.shape[0]))
            )
            .reset_index(level=[0, 1], drop=True)
        )

    return tweets


################################
# Old methods used in the deprecated notebooks
################################


def sample_legislator_tweets(
    db_engine: Engine,
    screen_name: str,
    db_table: str = "tweets",
    text_col: str = "text",
) -> pd.DataFrame:

    contexts_df = pd.read_sql_query(
        f"SELECT * FROM {db_table} \
                        WHERE screen_name = '{screen_name}' \
                        ORDER BY RANDOM() ",
        db_engine,
    )
    contexts_df.rename(columns={text_col: "text"}, inplace=True)
    logging.info(f"Sample for {screen_name}: {contexts_df.shape[0]}")

    return contexts_df


def compute_all_polarity(
    contexts_df: pd.DataFrame,
    polarity_ref: Dict,
    replace_nan: bool = False,
    nlp: lang.en.English = nlp,
):
    """
    Computes all 3 types of scores for all of a legislator's tweets
    (saves time when we resample during bootstrapping)
    replace_nan handles behavior of mean_per_scored_word when there are
    no scored words in a document: replaces the mean with 0
    """

    total_scores = []
    mean_per_word_scores = []
    mean_per_scored_word_scores = []

    with nlp.disable_pipes("tagger", "ner", "parser"):
        for tweet in contexts_df.text:
            tokenized = [t.text for t in nlp(tweet)]

            token_scores = [
                polarity_ref.get(token, None) for token in tokenized
            ]
            # mean_per_scored_word : need to handle [] and [None]
            nones_dropped = [i for i in token_scores if i]
            if replace_nan and not nones_dropped:
                # check if nones_dropped is []
                # replace mean([]) with 0.0
                mean_per_scored_word_scores.append(0.0)
            else:
                # the actual mean, or
                # NaN if mean([]) and replace_nan is False
                # mean([]) will still raise a RunTime warning that
                # can safely be ignored
                mean_per_scored_word_scores.append(np.mean(nones_dropped))

            # mean_per_word and total: replace None with 0
            token_scores_zero_out_nans = [i if i else 0 for i in token_scores]
            total_scores.append(sum(token_scores_zero_out_nans))
            mean_per_word_scores.append(
                np.mean(token_scores_zero_out_nans, dtype=np.float64)
            )

    contexts_df["total"] = total_scores
    contexts_df["mean_per_word"] = mean_per_word_scores
    contexts_df["mean_per_scored_word"] = mean_per_scored_word_scores
    return contexts_df


def bootstrap_precomputed_polarity(
    scored_df: pd.DataFrame,
    boot_size: int,
    n_resamples: int,
    with_replacement: bool = True,
) -> Dict:
    """
    When computing summary statistics over each SAMPLE of tweets
    we drop NaN tweets
    """
    if boot_size > scored_df.shape[0]:
        logging.info(
            "Warning: n requested > data size. Force sample with replacement."
        )
        with_replacement = True

    raw_stats = {"total": [], "mean_per_word": [], "mean_per_scored_word": []}

    for i in range(n_resamples):
        boot_df = scored_df.sample(
            n=boot_size,
            replace=with_replacement,
            random_state=i,  # must be different
        )

        raw_stats["total"].append(np.nanmean(boot_df["total"].to_numpy()))
        raw_stats["mean_per_word"].append(
            np.nanmean(boot_df["mean_per_word"].to_numpy())
        )
        raw_stats["mean_per_scored_word"].append(
            np.nanmean(boot_df["mean_per_scored_word"].to_numpy())
        )

    boot_stats = {}
    for k, v in raw_stats.items():
        boot_stats[k] = {"mean": np.mean(v), "std": np.std(v)}

    boot_stats["raw_stats"] = raw_stats
    boot_stats["sample_size"] = scored_df.shape[0]

    return boot_stats


def compute_tweet_polarity(
    tweet: str,
    polarity_ref: Dict,
    scoring: Literal["mean_per_word", "mean_per_scored_word", "total"],
    replace_nan: bool = False,
    nlp: lang.en.English = nlp,
):
    """
    scoring options include:
    "mean_per_word" = mean polarity per ALL words (words not in
    the polarity vocab are assigned 0)
    "mean_per_scored_word" = mean polarity per word in the polarity vocab
    "total" = sum over the tweet
    if replace_nan, NaN's are replaced with zero (this occurs when option
    mean_per_scored_word is selected but the tweet contains zero words with
    polarity scores)
    """
    with nlp.disable_pipes("tagger", "ner", "parser"):
        tokenized = [t.text for t in nlp(tweet)]
    if scoring == "mean_per_scored_word":
        polarity_scores = (
            polarity_ref.get(token, None) for token in tokenized
        )
        polarity_scores = [p for p in polarity_scores if p]
    else:
        polarity_scores = [polarity_ref.get(token, 0) for token in tokenized]

    if scoring == "total":
        return sum(polarity_scores)
    else:
        if replace_nan and not polarity_scores:
            return 0.0  # return 0 instead of np.mean([]) = NaN
        else:
            return np.mean(polarity_scores, dtype=np.float64)


def compute_sample_polarity(
    sample_df: pd.DataFrame,
    polarity_ref: Dict,
    scoring: Literal["mean_per_word", "mean_per_scored_word", "total"],
    replace_nan: bool = False,
) -> Dict:
    """
    Compute summary statistics for one sample of tweets
    for mean_per_scored_word option, a tweet that does not contain any
    scored words has a value of NaN, and it is dropped from the sample
    mean computation
    """
    scores = [
        compute_tweet_polarity(
            tweet,
            polarity_ref=polarity_ref,
            scoring=scoring,
            replace_nan=replace_nan,
        )
        for tweet in sample_df.text.tolist()
    ]

    results = {
        "sample_mean": np.nanmean(scores, dtype=np.float64),
        "std": np.std(scores),
        "scoring": scoring,
    }

    return results
