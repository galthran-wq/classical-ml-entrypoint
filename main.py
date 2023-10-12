import joblib
import itertools
from typing import List

import fire
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import gamma, uniform
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score



from src.data import BaseTransformPipeline, TextTransformPipeline

imp_params = [
    {
        "ts__num__impute": [SimpleImputer()],
        "ts__num__impute__strategy": ["median"],
    },
    {
        "ts__num__impute": [IterativeImputer()],
        "ts__num__impute__estimator": [
            # default
            BayesianRidge(),
            # todo: what else to add?
        ],
    },
    {
        "ts__num__impute": [KNNImputer()],
        "ts__num__impute__n_neighbors": [3, 5, 10],
    },
]

clf_params = [
    {
        "clf": [KNeighborsClassifier()],
        "clf__n_neighbors": [3, 5, 7, 10, 15],
        "clf__weights": ["uniform", "distance"]
    },
    {
        "clf": [RandomForestClassifier()],
        "clf__n_estimators": [50, 100, 200, 400],
        "clf__max_depth": [25, 50, 100],
    },
    {
        "clf": [SVC()],
        "clf__C": gamma(a=3.0),
        "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
    },
    {
        "clf": [GradientBoostingClassifier()],
        "clf__learning_rate": [0.1],
        "clf__n_estimators": [50, 100, 200, 300, 500, 1000],
        "clf__subsample": [1, 0.5],
        "clf__max_depth": [2, 3, 6],
        "clf__n_iter_no_change": [None, 5],
    },
    {
        "clf": [LogisticRegression()],
        "clf__penalty": ['elasticnet'],
        "clf__l1_ratio": uniform(0, 1),
        "clf__solver": ["saga"],
        "clf__C": gamma(a=3.0),
    },
    
]

grid_params = [{
    **imp_i_params,
    **clf_i_params,
} for imp_i_params, clf_i_params in 
    itertools.product(imp_params, clf_params)]


def get_transform_pipeline(str):
    if str == "text":
        return TextTransformPipeline
    else:
        return BaseTransformPipeline


def main(
    data_path: str, 
    output_path: str,
    label_col: str,
    # TODO: we should accept BaseTransformPipeline
    # to do that replace fire; add configs and jsonargparse cli
    cat_vars: List[str] = [], 
    num_vars: List[str] = [],
    text_vars: List[str] = [],
    transform_pipeline="base",
    #
):
    data = pd.read_csv(data_path)
    X = data.loc[:, cat_vars + num_vars + text_vars + [label_col]]
    X, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y = X[label_col]
    y_test = X_test[label_col]

    transform_pipeline = get_transform_pipeline(transform_pipeline)(
        cat_vars=cat_vars, 
        num_vars=num_vars,
        text_vars=text_vars
    ).get_pipeline()

    model_pipeline = Pipeline([
        ("ts", transform_pipeline),
        ("clf", LogisticRegression())
    ])

    boost_gs = RandomizedSearchCV(
        model_pipeline, grid_params, 
        random_state=42,
        n_iter=500,
        cv=5,
        verbose=2,
        error_score='raise',
        scoring=["f1", "recall", "precision", "accuracy"],
        refit="f1",
        n_jobs=4,
    ).fit(X, y)

    test_predict = boost_gs.predict(X_test)

    print(f"F1 = {f1_score(y_test, test_predict)}")
    print(f"Accuracy = {accuracy_score(y_test, test_predict)}")

    joblib.dump(boost_gs, output_path)


if __name__ == "__main__":
    fire.Fire(main)