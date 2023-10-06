from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseTransformPipeline:
    def __init__(self, *args, cat_vars: List[str], num_vars = List[str], **kwargs) -> None:
        self.enc_pipeline = ColumnTransformer([
            ("num", self.get_num_pipeline(), num_vars),
            ("cat", self.get_cat_pipeline(), cat_vars),
        ], remainder="drop")

    def get_num_pipeline(self):
        return Pipeline([
            ("impute", SimpleImputer()),
            ("scale", MinMaxScaler()),
        ])
    
    def get_cat_pipeline(self):
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


class TextTransformPipeline(BaseTransformPipeline):
    def __init__(self, text_cols: List[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enc_pipeline.transformers.append(
            ("text", self.get_text_pipeline(), text_cols[0])
        )

    def get_text_pipeline(self):
        text_prep = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english", dtype="float32")),
            # ("unsparse", FunctionTransformer(lambda x: x.toarray())),
        #     ("embed", PCA(n_components=100)),
        ])
        return text_prep