from typing import List

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils import PandasColumnTransformer
from ..const import NUM_KEY, CAT_KEY, FE_KEY
from ..grid import GridInstance, KEYS


class BaseTransformPipeline:
    def __init__(self, *args, cat_vars: List[str], num_vars: List[str], text_vars: List[str] = [], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_pipeline = PandasColumnTransformer([
            (NUM_KEY, self.setup_num_pipeline(), num_vars),
            (CAT_KEY, self.setup_cat_pipeline(), cat_vars),
        ], remainder="drop")
    
    def get_pipeline_by_key(self, key) -> Pipeline:
        for transformer_key, pipeline, _ in self.enc_pipeline.transformers:
            if transformer_key == key:
                return pipeline

    def setup_num_pipeline(self):
        return Pipeline([
            ("impute", SimpleImputer()),
            ("scale", MinMaxScaler()),
        ])
    
    def setup_cat_pipeline(self):
        return Pipeline([ 
            ("ohe", OneHotEncoder(
                # o.w. the data matrix is singular
                drop="first",
                handle_unknown="ignore",
                # o.w. raises error
                sparse=False,
            )),
        ])

    def get_pipeline(self):
        return self.enc_pipeline
    
    def set_params(self, params):
        self.enc_pipeline.set_params(params)
    
    def set_params_from_grid_instance(self, instance: GridInstance):
        for key in KEYS:
            key_grid = instance.get_by_key(key)
            key_pipeline = self.get_pipeline_by_key(key)
            if key_pipeline and key_grid:
                key_pipeline.set_params(**key_grid)


class TextTransformPipeline(BaseTransformPipeline):
    def __init__(self, *args, text_vars: List[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_pipeline.transformers.append(
            ("text", self.get_text_pipeline(), text_vars[0])
        )

    def get_text_pipeline(self):
        text_prep = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english", dtype="float32")),
            # ("unsparse", FunctionTransformer(lambda x: x.toarray())),
        #     ("embed", PCA(n_components=100)),
        ])
        return text_prep
