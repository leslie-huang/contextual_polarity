from joblib import load
import os
from typing import Optional, List
from sklearn.model_selection._search import GridSearchCV
import logging
import pandas as pd


def get_saved_classifier(
    results_root_dir: str, word: str
) -> Optional[GridSearchCV]:
    if word == word.lower():
        target_fn = word
    else:
        target_fn = word + "_"

    filenames = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(results_root_dir)
            for f in filenames
            if os.path.splitext(f)[1] == ".joblib"
        ]
    )
    grid = None
    for fn in filenames:
        if fn.split("/")[-1].split(".joblib")[0] == target_fn:
            logging.info(f"Loading from {fn}")
            grid = load(fn)

    return grid


def get_training_ids(
    results_root_dir: str, word: str, id_var: str = "id"
) -> List:
    """
    Get the IDs for the training data used to fit the classifier
    """
    if word == word.lower():
        target_fn = word
    else:
        target_fn = word + "_"

    filenames = sorted(
        [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(results_root_dir)
            for f in filenames
            if os.path.splitext(f)[1] == ".pkl"
        ]
    )
    results = None
    for fn in filenames:
        if fn.split("/")[-1].split(".pkl")[0] == target_fn:
            logging.info(f"Loading from {fn}")
            results = load(fn)

    if results:
        ids = [i[id_var] for i in results["metadata"]]
    else:
        ids = []
    return ids


def get_speakers_above_threshold(
    tweets: pd.DataFrame, official_only: bool = True, threshold: int = 40
) -> List:
    """
    Filters speakers who fall below the threshold
    """
    if official_only:
        counts = tweets[tweets.is_official == 1].screen_name.value_counts()
    else:
        counts = tweets.screen_name.value_counts()

    screen_names_above_cutoff = sorted(
        counts.index[counts >= threshold].tolist()
    )

    return screen_names_above_cutoff
