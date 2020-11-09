import pandas as pd
from sklearn import metrics
import lightgbm as lgb

def run(fold):
    df = pd.read_csv("../input/train_targets_folds.csv")

    df_train = df[df.kfold != fold].reset_index(drop=True)

if __name__ == "__main__":