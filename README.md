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
python main.py \
    --data_path ./heart.csv  \
    --label_col output --cat_vars [sex,cp,exan,ca,cap,fbs,rest_ech] \
    --num_vars [trtbps,chol,thalachh] \
    --output_path ./cache/heart_gs.pkl
```


## TODO

1. Flexible parameters
    - jsonargparse cli with run configs
2. feature engineering capabilities in data
3. visualizers and quick summaries
4. logging callbacks
