import pandas as pd


class BaseLoader:
    def __init__(self, label_col: str, data_path: str, to_split=False) -> None:
        self.data = pd.read_csv(data_path)
        feature_cols = self.data.columns.drop(label_col)
        self.X = self.data.loc[:, feature_cols]
        self.y = self.data.loc[:, label_col]
    
    def get_data(self):
        return self.X, self.y