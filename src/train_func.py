import setup
from configs.config import RANDOM_STATE, N_CHANNELS, SFREQ
from logger import logger
from utils import EEG, np, plt, tqdm
from collections import Counter
from datetime import datetime
import joblib
import sklearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pickle
import json
import argparse
import os  
import multiprocessing


def get_n_jobs():
    n_cpus = max(1, multiprocessing.cpu_count() // 2)
    return int(os.environ.get('SLURM_CPUS_PER_TASK', n_cpus))

def run(
    pipeline: sklearn.pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
    fname: str = None,
    model_path: str = None,
) -> np.ndarray:

    n_times = X_train.shape[-1]
    scores = []

    for t in range(n_times):
        X_train_t = X_train[:, :, t]
        X_val_t = X_val[:, :, t]

        pipeline.fit(X_train_t, y_train)
        y_pred = pipeline.predict(X_val_t)
        score = metrics.accuracy_score(y_val, y_pred)
        scores.append(score)
    logger.info(f"Fold: {fold}, Mean accuracy: {np.mean(scores):.3f}")

    if model_path:
        path = model_path / fname
        path.mkdir(exist_ok=True, parents=True)
        joblib.dump(pipeline, f"{path}/model_fold_{fold}.bin")

    return np.array(scores)

def train(
    n_splits: int,
    subject: np.ndarray,
    times: np.ndarray,
    fname: str,
    average: bool = False,
    random_state: int = RANDOM_STATE,
    N: int = None,
    reuse: bool = False,
    factor: float = None,
    output_dir: str = None,
    model_path: str = None,
    save_json: bool = False,
    save_image: bool = False,
) -> np.ndarray:
    
    n_jobs = get_n_jobs()
    
    logger.info(f"Number of cpus to run fold: {n_jobs}")

    X = subject["data"]
    y = subject["labels"]
    fname = subject["fname"]

    skf = StratifiedKFold(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=True
    )

    folds_split = dict()
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=X, y=y)):
        folds_split[fold +
                    1] = dict({"train_idx": train_idx, "val_idx": val_idx})

    labels_count = dict(Counter(y))
    logger.info(f"X shape: {X.shape} | y shape: {y.shape}")
    logger.info(f"Phonem: 'da' | label: 0 | Num: {labels_count[0]}")
    logger.info(f"Phonem: 'ba' | label: 1 | Num: {labels_count[1]}")

    def run_fold(fold:int)-> np.ndarray:
        logger.info(f"Running Fold {fold}.")
        val_indexes = folds_split[fold]["val_idx"]
        train_indexes = folds_split[fold]["train_idx"]

        X_train, X_val = X[train_indexes], X[val_indexes]
        y_train, y_val = y[train_indexes], y[val_indexes]

        if average:
            labels = subject["trials"].flatten()
            train_labels = labels[train_indexes]
            val_labels = labels[val_indexes]
            X_train, y_train = EEG(
                n_channels=N_CHANNELS,
                sfreq=SFREQ
            ).micro_average(
                X=X_train, labels=train_labels, N=N, reuse=reuse, factor=factor
            )

            X_val_len = len(X_val)
            if X_val_len // N < 10 and not reuse:
                raise ValueError(
                    f"Micro-average aborded for N={N}: too small data encountred!"
                )
            elif reuse and factor and (X_val_len * factor) + X_val_len < 10:
                raise ValueError(
                    f"Micro-average aborded for N={N}: too small data encountred!"
                )
            X_val, y_val = EEG(
                n_channels=N_CHANNELS,
                sfreq=SFREQ
            ).micro_average(
                X=X_val, labels=val_labels, N=N, reuse=reuse, factor=factor
            )

        pipeline_fold = sklearn.pipeline.Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
            ])

        start = datetime.now()
        score = run(
            pipeline=pipeline_fold,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fold=fold,
            model_path=model_path,
            fname=fname,
        )
        end = datetime.now()
        elapsed = (end - start).total_seconds()
        logger.info(f"Fold {fold}: start={start} -- end={end} -- running duration={elapsed:.4f} sec.")
        

        return score
        # scores.append(score)
    
    fold_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(run_fold)(fold) for fold in tqdm(folds_split, colour='BLUE', desc='Cross-val...') )
    fold_scores = np.array([score for score in fold_scores if score is not None])

    if save_json:
        save_results(
            scores=fold_scores,
            times=times,
            fname=fname,
            output_dir=output_dir,
            title='fold'
        )

    if save_image:
        plot_fold_history(
            scores=fold_scores,
            fname=fname+f"_{random_state}",
            times=times,
            output_dir=output_dir,
            plot=False
        )

    return np.mean(fold_scores, axis=0)

def save_results(
        scores: np.ndarray,
        times: np.ndarray,
        fname: str = None,
        output_dir: str = None,
        title: str = "Fold"):
    filename = f"{fname}_decoding_results.json"
    if output_dir:
        output_dir /= "vectors"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = output_dir / filename
    results = {f'{title}_{idx+1}': score.tolist() if isinstance(score,
                                                                np.ndarray) else score for idx, score in enumerate(scores)}
    results[f'{title}_mean'] = np.mean(scores, axis=0).tolist()
    results['times'] = times.tolist()

    try:
        with open(filename, 'w') as f:
            json.dump(results, f)
        logger.info(f'----- {title} results succesfully saved ------')
    except Exception as e:
        raise f"Error:  {e}"

def plot_fold_history(
    scores: np.ndarray,
    times: np.ndarray,
    fname: str = "",
    output_dir: str = None,
    plot: bool = False,
):
    _, ax = plt.subplots()

    for i, acc in enumerate(scores):
        ax.plot(times[: len(acc)], acc,
                label=f"fold: {i+1}", alpha=0.3, linewidth=1)

    mean_scores = np.mean(scores, axis=0)
    ax.plot(times[: len(mean_scores)], mean_scores, label="Mean accuracy")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title(f"Decoding over time (model performance of {fname})")
    ax.grid(True)
    if output_dir:
        output_dir /= "images"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = output_dir / f"{fname}_folds_decoding_results.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # logger.info('----- Folds accuracy graph succesfully saved ------')
    if plot:
        plt.show()

def plot_fold_loop_history(
    scores: np.ndarray,
    times: np.ndarray,
    fname: str = None,
    output_dir: str = None,
    plot: bool = False,
):

    _, ax = plt.subplots()

    for i, acc in enumerate(scores):
        if i == 0  or i == len(scores) - 1:
            ax.plot(times[: len(acc)], acc, label=f"iteration: {i+1}", alpha=0.3, linewidth=1)
        elif i == 1  and len(scores) > 2:
            ax.plot(times[: len(acc)], acc, label=f'.........', alpha=0.3, linewidth=1)
        else:
            ax.plot(times[: len(acc)], acc, alpha=0.3, linewidth=1)

    fname = fname.split('_')[0]

    mean_scores = np.mean(scores, axis=0)
    ax.plot(times[: len(mean_scores)], mean_scores, 
            label=f"Mean accuracy of {len(scores)} repetitions")
    
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title(f"Decoding over time ({fname} repeated {len(scores)} times)")
    ax.grid(True)
    if output_dir:
        output_dir /= "images"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = output_dir / f"{fname}_iterations_decoding_results.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # logger.info('----- Iterations accuracy graph succesfully saved ------')
    if plot:
        plt.show()

