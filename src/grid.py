import json
from typing import Dict, List, Any, Literal
from copy import deepcopy
from collections import defaultdict
import itertools

from sklearn.linear_model import (
    LogisticRegression,
    BayesianRidge,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import gamma, uniform
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler

from .const import (
    NUM_KEY,
    ORD_KEY,
    CAT_KEY,
    MODEL_KEY,
)
from .utils import str_to_class, class_to_str


Grid = List[Dict[str, Any]]

KEYS  = [ NUM_KEY, ORD_KEY, CAT_KEY, MODEL_KEY ]


class GridInstance:
    def __init__(self, instance: Dict[Any, Dict[str, Any]] = None) -> None:
        if instance is None:
            self.instance = {
                key: {}
                for key in KEYS
            }
        else:
            self.instance = instance

    @classmethod
    def from_dict(cls, instance_dict):
        instance = defaultdict(dict)
        for k, v in instance_dict.items():
            key, k = k.split("/")
            instance[key][k] = v
        return cls(instance)

    def to_dict(self):
        instance_dict = {}
        for key in KEYS:
            subinstance = self.get_by_key(key)
            if subinstance:
                for k, v in subinstance.items():
                    instance_dict[f"{key}/{k}"] = v
        return instance_dict
    
    def to_plain_dict(self):
        instance_dict = self.to_dict()
        for k, v in instance_dict.items():
            instance_dict[k] = repr(v)
        return instance_dict

    def get_by_key(self, key) -> Dict:
        return self.instance.get(key, None)

    def __repr__(self) -> str:
        return "GridInstance:\n" + json.dumps(self.to_plain_dict(), indent=2)


class BaseGrid:
    def __init__(self) -> None:
        self.grids: Dict[KEYS, Grid] = {
            key: []
            for key in KEYS
        }

    def get_grid_by_key(self, key) -> List[Dict]:
        return self.grids.get(key, None)

    def get_grid(self):
        grids: List[Grid] = []
        for key in KEYS:
            key_grid: Grid = deepcopy(self.get_grid_by_key(key))
            if key_grid:
                for i, case in enumerate(key_grid):
                    key_grid[i] = {
                        f"{key}/{k}": v
                        for k, v in case.items()
                    }
                grids.append(key_grid)

        grid_params = []
        for combination in itertools.product(*grids):
            combination_grid = {}
            for grid in combination:
                combination_grid.update(grid)
            grid_params.append(combination_grid)

        return grid_params
    
    def to_sweep(self) -> Dict[str, Any]:
        sweep_config = {}
        for key in KEYS:
            for k, v in self.get_grid_by_key(key).items():
                if k == MODEL_KEY:
                    v = class_to_str(v)
                sweep_config[f"{key}/{k}"] = v
        return sweep_config

    def from_sweep(self, sweep_config):
        for k, v in sweep_config.items():
            key, k = k.split("/")
            if k == MODEL_KEY:
                v = str_to_class(v)()
            self.grids[key][k] = v
        
    def get_instance(self) -> GridInstance:
        sampler = iter(ParameterSampler(self.get_grid(), n_iter=100))
        instance_dict = next(sampler)
        return GridInstance.from_dict(instance_dict)


class DefaultGrid(BaseGrid):
    def __init__(self) -> None:
        super().__init__()
        self.grids[MODEL_KEY] = [
            {
                f"{MODEL_KEY}": [KNeighborsClassifier()],
                f"{MODEL_KEY}__n_neighbors": [3, 5, 7, 10, 15],
                f"{MODEL_KEY}__weights": ["uniform", "distance"]
            },
            {
                f"{MODEL_KEY}": [RandomForestClassifier()],
                f"{MODEL_KEY}__n_estimators": [50, 100, 200, 400],
                f"{MODEL_KEY}__max_depth": [25, 50, 100],
            },
            {
                f"{MODEL_KEY}": [SVC()],
                f"{MODEL_KEY}__C": gamma(a=3.0),
                f"{MODEL_KEY}__kernel": ["linear", "poly", "rbf", "sigmoid"],
            },
            {
                f"{MODEL_KEY}": [GradientBoostingClassifier()],
                f"{MODEL_KEY}__learning_rate": [0.1],
                f"{MODEL_KEY}__n_estimators": [50, 100, 200, 300, 500, 1000],
                f"{MODEL_KEY}__subsample": [1, 0.5],
                f"{MODEL_KEY}__max_depth": [2, 3, 6],
                f"{MODEL_KEY}__n_iter_no_change": [None, 5],
            },
            {
                f"{MODEL_KEY}": [LogisticRegression()],
                f"{MODEL_KEY}__penalty": ['elasticnet'],
                f"{MODEL_KEY}__l1_ratio": uniform(0, 1),
                f"{MODEL_KEY}__solver": ["saga"],
                f"{MODEL_KEY}__C": gamma(a=3.0),
            },
        ]
        self.grids[NUM_KEY] = [
            {
                "impute": [SimpleImputer()],
                "impute__strategy": ["median"],
            },
            {
                "impute": [IterativeImputer()],
                "impute__estimator": [
                    # default
                    BayesianRidge(),
                    # todo: what else to add?
                ],
            },
            {
                "impute": [KNNImputer()],
                "impute__n_neighbors": [3, 5, 10],
            },
        ]