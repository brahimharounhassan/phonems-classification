import os
from logger import logger
from pathlib import Path
from typing import Tuple, Union
from pathlib import Path
import mne
import json
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
tqdm.pandas()
# import psutil

# os.environ["QT_QPA_PLATFORM"] = "xcb"
# import matplotlib
# matplotlib.use('TkAgg')

mne.set_log_level("WARNING")


class EEG(object):

    __event_id = {f"S_0{i+1}/DIN1": i+1 for i in range(8)}

    def __init__(
        self,
        n_channels: int = 128,
        sfreq: float = 250.0,
        plot_width: float = 18,
        plot_height: float = 6,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self._width = plot_width
        self._height = plot_height
        self.ch_names = [f"E{i+1:03}" for i in range(self.n_channels)]
        self.ch_types = ["eeg"] * self.n_channels
        self.info = mne.create_info(
            ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types
        )
        self.info["description"] = "EEG data for 'ba' and 'da' phonem's classification."

    def width(self, width) -> None:
        """Sets the plot width.

        Args:
          width (float): The new width value.
        """
        self._width = width

    def height(self, height) -> None:
        """Sets the plot height.

        Args:
          height (float): The new height value.
        """
        self._height = height

    width = property(None, width)
    height = property(None, height)

    def read_file(self, filename: str) -> dict:
        ext = Path(filename).suffix.lower()
        try:
            if ext == '.json':
                with open(filename, 'r') as f:
                    return json.load(f)
            elif ext == '.mat':
                return loadmat(filename)
        except Exception as e:
            logger.error(f"Error loading file {filename}: {e}")
            raise

    def get_subject_files(self, 
                          data_path: str,
                          condition: str,
                          ext: str = "mat"
                          ) -> dict:
        files = list()
        if "." not in ext:
            ext = f".{ext.strip().lower()}"
        pattern = f"{Path(data_path)}/{condition}*{ext}"
        files = list(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(
                "No matching files found!")
        logger.info(f"Total files found : {len(files)}")

        return {Path(f).stem: f for f in files}

    def get_subject(
        self, filename: str = None, time_points: int = 300, sfreq: float = 250.0
    ) -> dict:
        subject = self.read_file(filename)
        fname = Path(filename).stem

        if subject["data"].shape[-1] != time_points:
            subject["data"] = np.transpose(
                subject["data"], axes=(2, 0, 1)
            )
        subject["data"] = subject["data"].astype(np.float32)
        subject["fname"] = fname
        data = subject["data"]
        subject["volume"] = subject.pop("sourdine")
        subject["labels"] = subject["trials"]

        logger.info(f"Subject {fname} -> Data shape: {subject['data'].shape}")
        if np.isnan(subject['data']).any().item():
            logger.warning(
                f"Number of missing values: {np.isnan(subject['data']).sum()}")
        total_sample_points = len(subject["data"]) * time_points
        total_time = total_sample_points / sfreq
        logger.info(
            f"Total recording time: {total_time} sec ({total_time/60:.2f} min)"
        )
        
        return subject

    def get_vector(self, filename: str = None) -> dict:
        data = self.read_file(filename)
        basename = os.path.basename(filename).split("_")
        fname = basename[0]
        iteration = basename[1].split('repeated')[-1]
        data["fname"] = fname
        data["iteration"] = iteration

        return data

    def encode_labels(self, subject: dict) -> dict:
        labels_map = {i: 0 if i <= 4 else 1 for i in range(1, 9)}
        labels = subject["labels"].flatten()
        subject["labels"] = np.vectorize(labels_map.get)(labels)

        return subject

    def drop_trials(
        self,
        subject: dict,
        drop_trials: bool = False,
        include: list = None,
        drop_volume: bool = True,
        drop_fussy: bool = True,
    ) -> dict:

        include = [] if include is None else include
        labels = subject["labels"].flatten()
        volume = subject["volume"].flatten()
        fussy = subject["fussy"].flatten()
        final_indexes = np.array(np.arange(len(labels)))
        if drop_volume:
            final_indexes = np.where(volume >= 0.2)[0]

        if drop_fussy:
            fussy_indexes = np.isin(final_indexes, np.where(fussy != 1)[0])
            final_indexes = final_indexes[fussy_indexes]

        if drop_trials:
            if len(include) < 0:
                raise ValueError("The param 'include' must not be empty")
            
            trial_indexes = np.where(np.isin(labels, include))[0]
            trial_indexes = np.isin(final_indexes, trial_indexes)
            final_indexes = final_indexes[trial_indexes]

        subject["data"] = subject["data"][final_indexes]
        for key in ["labels", "onsets", "fussy", "volume", "sleep", "trials"]:
            subject[key] = subject[key].flatten()[final_indexes]
            
        return subject

    def _group_and_average(
        self, X: np.ndarray, indexes: np.ndarray, N: int, label=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        length = len(indexes) - (len(indexes) % N)
        indexes = indexes[:length]

        groups = [indexes[i: i + N] for i in range(0, length, N)]
        X_averaged = [np.mean(X[g], axis=0) for g in groups]
        y = [label] * len(groups)

        return np.array(X_averaged), np.array(y)

    def micro_average(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        N: int = 2,
        reuse: bool = False,
        factor: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_tmp, y_tmp = list(), list()

        logger.info(f"Applying micro-average of {N}.")
        logger.info(f"Shape before micro-average: {X.shape}.")
        for label in np.unique(labels):
            indexes = np.where(labels == label)[0]
            if reuse:
                size = len(indexes)
                if factor is not None:
                    size = round(len(indexes) * factor) + len(indexes)

                size *= N
                indexes = np.random.choice(a=indexes, size=size, replace=True)
            np.random.shuffle(indexes)
            X_averaged, y = self._group_and_average(
                X, indexes, N, label)

            X_tmp.extend(X_averaged)
            y_tmp.extend(y)

        y_tmp = [0 if y <= 4 else 1 for y in y_tmp]

        indexes = np.random.permutation(len(X_tmp))
        logger.info(f"Shape after micro-average: {np.shape(X_tmp)}.")

        return np.array(X_tmp)[indexes], np.array(y_tmp)[indexes]

    def to_mne(self, subject: dict, time_points: int = 300) -> dict:

        tmp_subject = subject
        data_3d = tmp_subject["data"]
        if data_3d.shape[-1] == time_points:
            data_3d = data_3d.transpose((1, 2, 0))
        if data_3d.shape[0] != self.n_channels:
            raise Exception("Error: inconsistent number of channels!")
        t = subject["t"].flatten()
        if self.sfreq is None:
            self.sfreq = 1 / np.diff(t).mean()
        onsets = tmp_subject["onsets"].flatten()
        labels = tmp_subject["labels"].flatten()
        data_2d = data_3d.reshape(self.n_channels, -1, order="F")
        durations = np.diff(onsets)
        durations = np.append(durations, durations[-1])
        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=[str(label) for label in labels],
        )
        onset_samples = ((onsets - onsets[0]) * self.sfreq).astype(int)
        events = np.column_stack(
            [onset_samples, np.zeros_like(
                labels, dtype=int), labels.astype(int)]
        )
        logger.info('Converting data to "mne RawArray".')
        raw = mne.io.RawArray(data_2d * 1e-6, self.info, copy="both").set_annotations(
            annotations
        )
        tmp_subject["data"] = raw
        tmp_subject["events"] = events
        tmp_subject["events_id"] = self.__event_id
        tmp_subject["annotations"] = annotations
        return tmp_subject

    def to_epochs(self, subject: dict) -> mne.EpochsArray:

        data_3d = subject
        onsets = data_3d["onsets"].flatten()
        labels = data_3d["labels"].flatten()
        t = data_3d["t"].flatten()
        duration = t[-1] - t[0]
        annotations = mne.Annotations(
            onset=onsets,
            duration=[duration] * len(onsets),
            description=[str(label) for label in labels],
        )
        onset_samples = ((onsets - onsets[0]) * self.sfreq).astype(int)
        events = np.column_stack(
            [onset_samples, np.zeros_like(
                labels, dtype=int), labels.astype(int)]
        )
        logger.info("Converting to epochs 3d array.")

        epochs = mne.EpochsArray(
            data_3d["data"] * 1e-6,
            self.info,
            events=events,
            tmin=t[0],
            event_id=self.__event_id,
        ).set_annotations(annotations)
        return epochs

    def plot_events(
        self, raw: dict, title: str, plot: bool = False, output_path: str = None
    ) -> None:

        events = raw["events"]
        fig = mne.viz.plot_events(
            events=events, sfreq=self.sfreq, event_id=self.__event_id, show=False
        )
        fig.set_size_inches(self._width, self._height)
        fig.canvas.manager.set_window_title(title)
        title = "_".join(title.lower().split(" "))
        plt.title(title)

        if output_path:
            output_path /= "visualization/"
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f"{title}.png",
                        dpi=300, bbox_inches="tight")
        if plot:
            plt.show(block=False)
            plt.pause(0.1)

    def plot_signals(
        self,
        raw_subject: mne.io.Raw,
        title: str,
        channels: list = None,
        duration: float = 10,
        plot: bool = False,
        output_path: str = None,
    ) -> None:
        events = raw_subject["events"]
        data = raw_subject["data"]
        if channels is None:
            fig = data.plot(
                events=events,
                duration=duration,
                n_channels=len(raw_subject.ch_names),
                show_scrollbars=False,
                show=False,
            )
        else:
            fig = (
                data
                .pick(channels)
                .plot(
                    events=events,
                    duration=duration,
                    n_channels=len(channels),
                    show_scrollbars=False,
                    show=False,
                )
            )
        fig.canvas.manager.set_window_title(title)
        fig.set_size_inches(self._width, self._height)
        title = "_".join(title.lower().split(" "))
        plt.title(title)

        if output_path:
            output_path /= "visualization/"
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f"{title}.png",
                        dpi=300, bbox_inches="tight")
        if plot:
            plt.show(block=False)
            plt.pause(0.1)

    def plot_epochs(
        self,
        subject_epochs: mne.EpochsArray,
        title: str,
        channels: list = None,
        n_epochs: int = 20,
        plot: bool = False,
        output_path: str = None,
    ) -> None:
        events = subject_epochs.events
        if channels is None:
            fig = subject_epochs.plot(
                n_epochs=n_epochs, events=events, show=False, show_scrollbars=False
            )
        else:
            fig = subject_epochs.plot(
                picks=channels,
                n_epochs=n_epochs,
                events=events,
                show_scrollbars=False,
                show=False,
            )
        fig.canvas.manager.set_window_title(title)
        title = "_".join(title.lower().split(" "))
        plt.title(title)
        fig.set_size_inches(self._width, self._height)
        if output_path:
            output_path /= "visualization/"
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f"{title}.png",
                        dpi=300, bbox_inches="tight")
        if plot:
            plt.show(block=False)
            plt.pause(0.1)


def get_file_name(subject_id:str, subject_files:dict)->str:
    if subject_id not in subject_files: 
        logger.info(f"Available subjects: \n{np.sort(list(subject_files.keys()))}")
        raise ValueError("Inexistant subject: specify a correct subject's number among top list:")
    logger.info(f'Subject {subject_id} selected')
    
    return subject_files[subject_id]