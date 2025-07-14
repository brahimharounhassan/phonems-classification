import src.setup
from src.utils import EEG
import numpy as np
import pytest
from configs.config import *


eeg = EEG(
    n_channels=N_CHANNELS, sfreq=SFREQ, plot_height=6, plot_width=10
)


def test_eeg_initialization():
    assert eeg.n_channels == 128
    assert eeg.sfreq == 250.0
    assert len(eeg.ch_names) == 128
    assert all(ch.startswith("E") for ch in eeg.ch_names)


def test_load_subject():
    subject_files = eeg.get_subject_files(
        CLEAN_SUBJECTS_PATH, ext="mat"
    )
    filename = subject_files[0]
    subject = eeg.get_subject(filename, N_TIMES, SFREQ)
    assert subject["data"].shape[0] > 0
    assert subject["data"].ndim == 3


@pytest.fixture
def fake_subject():
    """Create a generic dataset with (10 epochs, 128 channels, 300 time-points)."""
    return {
        "data": np.random.randn(10, 128, 300),
        "labels": np.array([1, 2, 1, 3, 7, 5, 8, 6, 4, 6]),
        "volume": np.array([0.3, 0.8, 0.25, 0.15, 0.35, 0.10, 1.0, 0.015, 0.18, 0.05]),
        "fussy": np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1]),
        "sleep": np.zeros(10),
        "onsets": np.arange(10),
        "t": np.linspace(-0.2, 0.9, 300),
        "fname": "test01",
    }


def test_encode_labels(fake_subject):
    labeled = eeg.encode_labels(fake_subject)
    assert np.array_equal(labeled["labels"], [0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    assert labeled["labels"].shape[0] == labeled["data"].shape[0]


def test_drop_trials(fake_subject):
    result = eeg.drop_trials(
        fake_subject,
        drop_trials=True,
        include=[1, 2, 7, 8],
        drop_volume=True,
        drop_fussy=True,
    )
    assert len(result["labels"]) == 5
    assert all([label in [1, 2, 7, 8] for label in result["labels"]])


def test_micro_average(fake_subject):
    averaged_data, labels = eeg.micro_average(
        fake_subject["data"], fake_subject["labels"], N=2, reuse=False
    )

    assert len(averaged_data) == 2
    assert averaged_data.shape[1:] == (128, 300)
    assert averaged_data.shape[0] == labels.shape[0]


def test_group_and_average(fake_subject):
    label = 6
    X = fake_subject["data"]
    y = fake_subject["labels"]
    indexes = np.where(y == label)[0]
    N = 2

    averaged_data, y = eeg._group_and_average(X, indexes, N, label)

    assert len(averaged_data) == 1
    assert len(y) == 1
    assert averaged_data.shape[0] == y.shape[0]
    assert averaged_data.shape[1:] == (128, 300)
