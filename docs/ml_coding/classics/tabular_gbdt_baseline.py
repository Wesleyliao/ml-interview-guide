import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


def run_tabular_baseline(df, target_col="target"):
    # 1. Quick data check & split FIRST (use a time cutoff instead if forecasting!).
    print("shape:", df.shape, "pos rate:", df[target_col].mean())
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 2. Fastest categorical + missing handling: cast strings to 'category'.
    # LightGBM handles NaNs and 'category' columns natively—no imputer or one-hot encoder needed.
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    # 3. Fit barebones LightGBM baseline.
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)

    # 4. Evaluation (compare train vs. test to spot overfitting, plus precision/recall/F1/AUC).
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary", zero_division=0
    )
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    print("train acc:", round(train_accuracy, 4), "test acc:", round(test_accuracy, 4))
    print("precision:", round(precision, 4), "recall:", round(recall, 4), "f1:", round(f1, 4), "auc:", round(test_auc, 4))
    return clf


# Where to iterate next once this baseline runs end-to-end:
# - Overfitting (train_acc >> test_acc): add early_stopping on a val split, lower num_leaves / max_depth.
# - Leakage check: print sorted zip(X_train.columns, clf.feature_importances_) to catch suspiciously perfect columns.
# - Feature ideas: add 2-3 domain ratios, recency deltas, or train-only group aggregates.
# - Imbalanced classes: tune threshold on clf.predict_proba(X_val)[:, 1] instead of default 0.5.
