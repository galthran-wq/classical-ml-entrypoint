from typing import List, Union, Dict, Any

import wandb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .grid import GridInstance


class BaseLogger:

    def log(grid: Union[GridSearchCV, RandomizedSearchCV]):
        pass

    def on_run_start(self, instance: GridInstance):
        pass

    def on_run_end(self, metrics: Dict[str, float]):
        pass


class WandbLogger(BaseLogger):
    def __init__(self, project: str, name: str):
        super().__init__()
        wandb.login()
        self.project = project
        self.name = name
    
    def on_run_start(self, instance: GridInstance):
        wandb.init(project=self.project, name=self.name, config=instance.to_plain_dict())
    
    def on_run_end(self, metrics: Dict[str, float]):
        wandb.log(metrics)
        wandb.finish(0)
