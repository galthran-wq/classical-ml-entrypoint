from typing import Any, List

import wandb
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

from .data import (
    BaseLoader, BaseTransformPipeline
)
from .model import BaseModel
from .grid import BaseGrid, GridInstance
from .logger import BaseLogger


class Pipeline:
    def __init__(
        self,
        scoring: List[str],
        loader: BaseLoader,
        transformer: BaseTransformPipeline,
        grid: BaseGrid,
        logger: BaseLogger,
        model: BaseModel = None,
        cv: int = 5,
        n_iter = 100,
    ) -> None:
        self.loader = loader
        self.transformer = transformer
        self.model = model
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.grid = grid
        self.logger = logger

        if model is None:
            self.model = BaseModel()

    def run_iter(self, *args: Any, **kwds: Any) -> Any:
        instance: GridInstance = self.grid.get_instance()
        self.logger.on_run_start(instance)

        X, y = self.loader.get_data()
        pipeline = self.get_pipeline(instance)
        results = cross_validate(
            pipeline, 
            X, y, 
            cv=self.cv, 
            scoring={
                k: k
                for k in self.scoring
            }
        )
        results = self.get_cv_metrics(results)
        self.logger.on_run_end(results)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for i in range(self.n_iter):
            self.run_iter()
    
    def get_pipeline(self, instance):
        self.transformer.set_params_from_grid_instance(instance)
        self.model.set_params_from_grid_instance(instance)
        transformer = self.transformer.get_pipeline()
        model = self.model.get_pipeline()
        pipeline = make_pipeline(transformer, model)
        return pipeline

    def get_cv_metrics(self, cv_results):
        return {
            f"cv_{k}": np.mean(cv_results[f"test_{k}"])
            for k in self.scoring
        }
