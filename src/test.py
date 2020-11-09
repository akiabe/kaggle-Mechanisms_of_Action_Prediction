from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

X = np.array([[1,2], [3,4], [1,2], [3,4], [1,2], [3,4], [1,2], [3,4]])
y = np.array([[0,0], [0,0], [0,1], [0,1], [1,1], [1,1], [1,0], [1,0]])

#print(X)
#print(y)

mskf = MultilabelStratifiedKFold(n_splits=2)
for train_idx, valid_idx in mskf.split(X, y):
    print(f"train: {train_idx}, valid: {valid_idx}")
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]