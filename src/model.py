from sklearn.ensemble import GradientBoostingClassifier as _GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from .const import MODEL_KEY


class BaseModel:
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            (MODEL_KEY, LogisticRegression())
        ])
    
    def set_params(self, params):
        self.pipeline.set_params(params)
    
    def get_pipeline(self):
        return self.pipeline
    
    