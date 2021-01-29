import pandas as pd
from tweetbert.embedded.word_dataset import WordDataset
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import logging

gridsearch_params = {"cv": 5, "n_jobs": -1, "scoring": "accuracy"}

classifiers = {"SVM_rbf": SVC(), "SVM_linear": SVC()}

# pipeline options
pipeline_params = {
    "SVM_rbf": [
        {
            "classify__kernel": ["rbf"],
            "classify__gamma": ["scale", "auto"],  # for rbf
            "classify__C": [1, 10, 100, 1000],
            "classify__random_state": [1],
            "classify__probability": [True],
        }
    ],
    "SVM_linear": [
        {
            "classify__kernel": ["linear"],
            "classify__C": [1, 10, 100, 1000],
            "classify__random_state": [1],
            "classify__probability": [True],
            "classify__max_iter": [5000],
        }
    ],
}


def gridsearch_one_word(
    clf_dataset: WordDataset,
    classifiers: Dict,
    pipeline_params: Dict,
    gridsearch_params: Dict,
) -> tuple:
    """
    Performs a gridsearch over multiple pipelines, one for each
    classifier x pipeline combination. Combines the results
    and returns them and the predictions from the best model
    from each gridsearch
    """
    results_dfs = []
    models_predictions = (
        []
    )  # stores the predictions from the best model from each pipeline
    best_grid = None
    best_score = None

    for classifier_name, classifier in classifiers.items():
        logging.info(f"Using {classifier_name}")
        pipes = [("scaler", StandardScaler()), ("classify", classifier)]

        pipeline = Pipeline(pipes)

        grid = GridSearchCV(
            pipeline,
            param_grid=pipeline_params[classifier_name],
            **gridsearch_params,
            # verbose=2,
        )
        grid.fit(clf_dataset.pooled_X, clf_dataset.labels)

        df = pd.DataFrame.from_dict(grid.cv_results_)
        df["classifier"] = classifier_name

        models_predictions.append(
            {
                "classifier": classifier_name,
                "predictions": grid.predict(clf_dataset.pooled_X),
            }
        )

        results_dfs.append(df)

        # keep track of whether the radial or linear grid is the one we save
        if best_score is None:
            logging.info(f"Initializing best_score to {grid.best_score_}")
            best_score = grid.best_score_
            best_grid = grid
        if best_score < grid.best_score_:
            logging.info(f"Updating best_score to {grid.best_score_}")
            best_score = grid.best_score_
            best_grid = grid

    combined_best = pd.concat(results_dfs, ignore_index=True, sort=True)

    return (
        {
            "cv_results": combined_best,
            "models_predictions": models_predictions,
            "metadata": clf_dataset.metadata,
            "true_labels": clf_dataset.labels,
            "word": clf_dataset.word,
        },
        best_grid,
    )
