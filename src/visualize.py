import setup
from configs.config import *
from utils import (EEG, mne, np, plt, get_file_name)
from logger import logger
import argparse


def plot_conditions(
    subject: dict, title: str, plot: bool = False, output_path: str = None
):
    # onsets = subject["onsets"].flatten()
    labels = subject["labels"].flatten()
    volume = subject["volume"].flatten()
    fussy = subject["fussy"].flatten()
    sleep = subject["sleep"].flatten()
    # times = onsets - onsets[0]
    times = np.arange(len(labels)) * N_TIMES
    plt.figure(figsize=(12, 4))
    # plt.plot(times, labels / 8, label="labels")
    plt.plot(times, sleep, label="sleep")
    plt.plot(times, fussy, label="fussy")
    plt.plot(times, volume, label="volume")

    labels = [f"{times[0]}", f"{times[-1]}"]
    ticks = [times[0], times[-1]]
    plt.xticks(ticks=ticks, labels=labels)

    plt.title(title)
    plt.ylabel("Value")
    plt.xlabel("Sampled points")
    plt.legend()
    plt.tight_layout()
    if output_path:
        output_path /= "visualization/"
        output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path / f"{title.lower()}.png",
                    dpi=300, bbox_inches="tight")

    if plot:
        plt.show(block=False)
        plt.pause(0.1)


if __name__ == "__main__":
    # SETTING UP CLASSES INSTANCIATION

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_raw', type=str, required=True, default=CLEAN_SUBJECTS_PATH,
                        help='Raw dataset path')
    
    parser.add_argument('--data_clean', type=str, required=True, default=CLEAN_SUBJECTS_PATH,
                        help='Clean dataset path')

    parser.add_argument('--output_dir', type=str, required=True, default=OUTPUT_PATH,
                        help='Output path for results.')

    parser.add_argument('--subject', type=str, required=True, default=3,
                        help='Subject num.')

    args = parser.parse_args()

    eeg = EEG(
        n_channels=N_CHANNELS,
        sfreq=SFREQ
    )

    # DATA READING
    # CLEAN DATA
    CLEAN_SUBJECTS_PATH = Path(args.data_clean)
    RAW_SUBJECTS_PATH = Path(args.data_raw)
    OUTPUT_PATH = Path(args.output_dir)

    clean_subject_files = eeg.get_subject_files(
        data_path=CLEAN_SUBJECTS_PATH,
        condition='/**/',
        ext=FILE_EXT
    )

    file_num = args.subject
    clean_filename = get_file_name(file_num, clean_subject_files)

    clean_subject = eeg.get_subject(
        filename=clean_filename, time_points=N_TIMES, sfreq=SFREQ
    )

    clean_mne_subject = eeg.to_mne(clean_subject.copy())
    clean_epochs_subject = eeg.to_epochs(clean_subject.copy())

    # RAW DATA
    raw_subject_files = eeg.get_subject_files(
        data_path=RAW_SUBJECTS_PATH,
        condition='/**/',
        ext=FILE_EXT
    )

    raw_filename = get_file_name(file_num, raw_subject_files)

    raw_subject = eeg.get_subject(
        filename=raw_filename, time_points=N_TIMES, sfreq=SFREQ
    )

    raw_mne_subject = eeg.to_mne(raw_subject.copy())
    raw_epochs_subject = eeg.to_epochs(raw_subject.copy())

    # PLOTTING CONTINUOUS AND EPOCHED SIGNALS, AND EVENTS
    # CLEAN DATA
    duration = 50
    channels_num = 20
    epochs_num = 30
    c_fname = clean_subject["fname"]
    channels = [f"E{i+1:03}" for i in range(channels_num)]

    eeg.plot_signals(
        raw_subject=clean_mne_subject,
        title=f"Clean signals of {c_fname}",
        channels=channels,
        duration=duration,
        plot=True,
        output_path=OUTPUT_PATH,
    )
    eeg.plot_epochs(
        clean_epochs_subject,
        title=f"Clean epoched signals of {c_fname}",
        channels=channels,
        n_epochs=epochs_num,
        plot=True,
        output_path=OUTPUT_PATH,
    )

    eeg.plot_events(
        clean_mne_subject,
        title=f"Clean data events of {c_fname}",
        plot=False,
        output_path=OUTPUT_PATH,
    )

    # RAW DATA
    r_fname = raw_subject["fname"]
    eeg.plot_signals(
        raw_subject=raw_mne_subject,
        title=f"Raw signals of {r_fname}",
        channels=channels,
        duration=duration,
        plot=False,
        output_path=OUTPUT_PATH,
    )

    eeg.plot_epochs(
        raw_epochs_subject,
        title=f"Raw epoched signals of {r_fname}",
        channels=channels,
        n_epochs=epochs_num,
        plot=False,
        output_path=OUTPUT_PATH,
    )

    eeg.plot_events(
        raw_mne_subject,
        title=f"Raw data events of {r_fname}",
        plot=False,
        output_path=OUTPUT_PATH,
    )

    # COMPARING ONE RAW AND CLEAN SIGNAL CHANNEL
    signal_raw = raw_subject["data"].copy()
    signal_raw = signal_raw.transpose((1, 2, 0)).reshape(
        N_CHANNELS, -1, order="F"
    )

    signal_clean = clean_subject["data"].copy()
    signal_clean = signal_clean.transpose((1, 2, 0)).reshape(
        N_CHANNELS, -1, order="F"
    )

    limit = 30000
    start = 123
    end = 126
    n_channels = end - start

    plot = False
    save = True
    fig, axes = plt.subplots(n_channels, 1, figsize=(
        20, 2 * n_channels), sharex=True)
    times = np.arange(limit) / SFREQ
    for i, channel in enumerate(range(start, end)):
        title = f"E{channel + 1:03}"
        ax = axes[i]
        ax.plot(times, signal_raw[channel][:limit], label="raw")
        ax.plot(times, signal_clean[channel][:limit], label="clean")
        ax.set_title(title)
        ax.set_ylabel("Amplitude (V)")
        ax.set_ylim([-80, 80])
    plt.legend(loc="best")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    if save:
        output_path = OUTPUT_PATH / "visualization/"
        output_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            output_path /
            f"clean_vs_raw_signals_comparison_of_({c_fname}).png",
            dpi=300,
            bbox_inches="tight",
        )
    if plot:
        plt.show(block=False)
        plt.pause(0.1)

    # PLOT ONSETS, TRIALS, VOLUME, FUSSY AND SLEEP ATTRIBUTES
    title = f"Conditions before processing of {c_fname}: (with all trials [1,2,3,4,5,6,7,8])"
    plot_conditions(clean_subject, title, False, OUTPUT_PATH)

    include = [1, 2, 7, 8]
    clean_subject = eeg.drop_trials(
        subject=clean_subject,
        drop_trials=True,
        include=include,
        drop_volume=True,
        drop_fussy=True,
    )
    title = f"Conditions after processing of {c_fname}: (with trials [1,2,7,8])"
    plot_conditions(clean_subject, title, False, OUTPUT_PATH)

    include = [1, 8]
    clean_subject = eeg.drop_trials(
        subject=clean_subject,
        drop_trials=True,
        include=include,
        drop_volume=True,
        drop_fussy=True,
    )
    title = f"Conditions after processing of {c_fname}: (with trials [1,8])"
    plot_conditions(clean_subject, title, False, OUTPUT_PATH)

    plt.show()
    # plt.close()
