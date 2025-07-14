import src.setup
from src.logger import logger, logging
from src.train_func import run, train
from configs.config import OUTPUT_PATH, MODEL_PATH
from unittest.mock import patch
import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@pytest.fixture(autouse=True)
def disable_logging():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


scaler = StandardScaler()
clf = LogisticRegression(solver="liblinear", max_iter=1000)

pipeline = Pipeline([("scaler", scaler), ("clf", clf)])


def test_run():
    X = np.random.rand(100, 128, 300)
    y = np.array([0] * 50 + [1] * 50)
    s = np.random.permutation(len(y))
    y = y[s]
    s = np.random.permutation(20)
    fold = 0
    folds_split = {0: {"train_idx": np.arange(
        70)[s], "val_idx": np.arange(70, 100)}}

    model_path = MODEL_PATH / "test"
    model_path.mkdir(exist_ok=True, parents=True)

    fname = "test01"
    fold = 0
    val_indexes = folds_split[fold]["val_idx"]
    train_indexes = folds_split[fold]["train_idx"]

    X_train, X_val = X[train_indexes], X[val_indexes]
    y_train, y_val = y[train_indexes], y[val_indexes]

    scores = run(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        fold=fold,
        model_path=model_path,
        fname=fname,
        verbose=False,
    )
    assert len(scores) == 300
    assert all(0 <= s <= 1 for s in scores)


@patch("train.run")
def test_train(mock_run):
    mock_run.return_value = np.array([0.7, 0.8, 0.9, 0.85, 0.75])

    subject = dict()
    subject["data"] = np.random.rand(100, 128, 300)
    y = np.array([0] * 50 + [1] * 50)
    s = np.random.permutation(len(y))
    subject["labels"] = y[s]
    subject["fname"] = "test01"

    skf = StratifiedKFold(n_splits=2)

    model_path = MODEL_PATH / "test/"
    model_path.mkdir(exist_ok=True, parents=True)
    output_dir = OUTPUT_PATH / "test/"

    train(
        pipeline=pipeline,
        kfold=skf,
        subject=subject,
        times=np.linspace(-0.2, 1.0, 300),
        plot=False,
        average=False,
        N=None,
        reuse=False,
        factor=None,
        model_path=model_path,
        output_dir=output_dir,
        verbose=False,
    )
    plt.show()
    plt.close()
    # assert mock_run.call_count == 2


def test_run_invalid_data_shape():
    X = np.random.rand(20, 5)
    y = np.array([0] * 10 + [1] * 10)

    folds_split = {
        0: dict(
            {
                "train_idx": np.random.randint(0, 9, size=10),
                "val_idx": np.random.randint(10, 21, size=10),
            }
        )
    }
    fname = "test01"
    fold = 0
    try:
        val_indexes = folds_split[fold]["val_idx"]
        train_indexes = folds_split[fold]["train_idx"]

        X_train, X_val = X[train_indexes], X[val_indexes]
        y_train, y_val = y[train_indexes], y[val_indexes]

        run(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fold=fold,
            model_path=None,
            fname=fname,
            verbose=False,
        )
    except (IndexError, ValueError):
        logger.error("Error: Expected an IndexError due to invalid shape")
