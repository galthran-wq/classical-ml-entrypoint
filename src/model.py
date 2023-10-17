from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .const import MODEL_KEY
from .grid import GridInstance


class BaseModel:
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            (MODEL_KEY, LogisticRegression())
        ])
    
    def set_params_from_grid_instance(self, instance: GridInstance):
        params = instance.get_by_key(MODEL_KEY)
        if params:
            self.pipeline.set_params(**params)
    
    def get_pipeline(self):
        return self.pipeline
    
    