import os
import pickle
import time

import pytest
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model_baseline.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    # 最新モデルを常に上書き
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # baseline がまだなければ一度だけ保存
    if not os.path.exists(BASELINE_MODEL_PATH):
        with open(BASELINE_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    return model, X_test, y_test


@pytest.fixture(scope="session")
def baseline_model_and_metrics(sample_data):
    """baseline モデルをロードし、メトリクスを事前計算"""
    if not os.path.exists(BASELINE_MODEL_PATH):
        pytest.skip(f"Baseline model not found at {BASELINE_MODEL_PATH}")

    with open(BASELINE_MODEL_PATH, "rb") as f:
        baseline_model = pickle.load(f)

    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    # 同じ分割を使う
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 精度
    y_pred = baseline_model.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred)

    # 推論時間
    start = time.time()
    baseline_model.predict(X_test)
    base_time = time.time() - start

    return X_test, y_test, base_acc, base_time


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.75, f"モデルの精度が低すぎます: {acc}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model
    start = time.time()
    model.predict(X_test)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"推論時間が長すぎます: {elapsed:.3f}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def build():
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

    m1, m2 = build(), build()
    m1.fit(X_train, y_train)
    m2.fit(X_train, y_train)

    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)

    assert np.array_equal(p1, p2), "モデルの予測結果に再現性がありません"


def test_performance_regression(train_model, baseline_model_and_metrics):
    """
    過去バージョンと比較して、
    - 精度が劣化していないこと
    - 推論時間が過大になっていないこと
    """
    model, X_test, y_test = train_model
    _, _, base_acc, base_time = baseline_model_and_metrics

    new_pred = model.predict(X_test)
    new_acc = accuracy_score(y_test, new_pred)
    assert (
        new_acc >= base_acc
    ), f"Accuracy regression: new={new_acc:.4f}, baseline={base_acc:.4f}"

    start = time.time()
    model.predict(X_test)
    new_time = time.time() - start
    # baseline の 1.2 倍まで許容
    assert (
        new_time <= base_time * 1.2
    ), f"Inference time regression: new={new_time:.3f}s, baseline={base_time:.3f}s"
