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


class Pipeline:
    def __init__(
        self,
        project: str,
        name: str,
        scoring: List[str],
        loader: BaseLoader,
        transformer: BaseTransformPipeline,
        grid: BaseGrid,
        model: BaseModel = None,
        cv: int = 5,
    ) -> None:
        self.loader = loader
        self.transformer = transformer
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.grid = grid

        self.project = project
        self.name = name

        if model is None:
            self.model = BaseModel()

        wandb.login()
        wandb.init(project=project, name=name)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        instance: GridInstance = self.grid.get_instance()

        X, y = self.loader.get_data()
        pipeline = self.get_pipeline()
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
    
    def get_pipeline(self):
        transformer = self.transformer.get_pipeline()
        model = self.model.get_pipeline()
        pipeline = make_pipeline(transformer, model)
        return pipeline

    def get_cv_metrics(self, cv_results):
        return {
            f"cv_{k}": np.mean(cv_results[f"test_{k}"])
            for k in self.scoring
        }
