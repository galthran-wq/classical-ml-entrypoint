Simple package to get some baselines using classical ML models.

Uses scikit-learn's models and pipeline mechanisms.

### Quick start

1. Get some data:

```sh
kaggle datasets download -d uciml/pima-indians-diabetes-database
unzip pima-indians-diabetes-database.zip
rm pima-indians-diabetes-database.zip
```

2. Run

```sh
python driver.py --config ./configs/pima/base.yaml
```


## TODO

1. feature engineering capabilities in data
