import setup
from configs.config import *
from logger import logger
from utils import EEG, np, plt
from utils import get_file_name
import argparse
from pathlib import Path

def plot_scores(scores, times, title, save_path=None, experiment='', show_all=True):
    fig, ax = plt.subplots()
    mean_scores = np.mean(list(scores.values()), axis=0)
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i) for i in range(len(scores))]

    for i, (label, acc) in enumerate(scores.items()):
        if show_all or i in [0, len(scores) - 1]:
            ax.plot(times[:len(acc)], acc, label=label, alpha=0.7, color=colors[i % len(colors)])
        elif i == 1 and len(scores) > 2:
            ax.plot(times[:len(acc)], acc, label='.........', alpha=0.5, linewidth=1)
        else:
            ax.plot(times[:len(acc)], acc, alpha=0.5, linewidth=1)

    ax.plot(times[:len(mean_scores)], mean_scores, label="Mean", color='blue')
    ax.axhline(0.5, color="k", linestyle="--", label="Chance")
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accuracy")
    # ax.set_ylim([.49,.53])
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True)

    if save_path:
        save_path.mkdir(exist_ok=True, parents=True)
        filename = save_path / f"{experiment}_decoding_performances.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info('----- Overall comparison graph successfully saved ------')

    # plt.show()

def load_scores(filename):
    data = EEG().get_vector(filename)
    data.pop("fname", None)
    data.pop("iteration", None)
    times = data.pop("times", None)
    mean_scores = data.pop("iteration_mean", None)
    return np.array(list(data.values())), np.array(mean_scores), times

def get_subject_key(subject_files, subject_id, N=None, n_iter=50, pattern=""):
    key_prefix = f"{subject_id}_average_N{N}_repeated{n_iter}" if N else f"{subject_id}_repeated{n_iter}"
    for key, path in subject_files.items():
        if key.startswith(key_prefix) and pattern in key:
            return path
    return None

def get_subjects_keys(subject_files, pattern):
    return [path for key, path in subject_files.items() if pattern in key]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--reuse', type=bool, default=False)
    parser.add_argument('--drop_t', type=bool, default=False)
    parser.add_argument('--include', type=str, default=None)
    parser.add_argument('--avg', type=bool, default=False)
    parser.add_argument('--N', type=str, default=None)
    parser.add_argument('--iter', type=int, required=True)
    parser.add_argument('--subject', type=str, required=True)
    args = parser.parse_args()

    eeg = EEG(n_channels=N_CHANNELS, sfreq=SFREQ)
    condition_parts = [
        f"/train/dropped_{'trials_' if args.drop_t else ''}volume_fussy/",
        "micro_avg/" if args.avg else f"no_avg_{args.iter}/",
        f"include_{args.include.replace(',', '_')}/" if args.include else "",
        f"{'reuse' if args.reuse else 'not_reuse'}_{args.iter}/" if args.avg else ""
    ]
    path = ''.join(condition_parts)

    OUTPUT_PATH = Path(args.output_dir + "/experimental_results/" + path.replace('/train', ''))
    DATA_PATH = Path(args.data_dir) / Path(path.split('/train/')[1])#/ Path(condition)

    all_scores =  {}

    if args.subject.lower() != "all":
        subject_id = f"S{int(args.subject):02d}"
        condition = f"**/{subject_id}"
        subject_files = eeg.get_subject_files(data_path=DATA_PATH, condition=condition, ext='json')
        if args.N and "," in args.N:
            start_n, end_n = map(int, args.N.split(','))
            for i in range(start_n, end_n):
                fname = get_subject_key(subject_files, subject_id, i, args.iter, f"_N{i}_")
                scores, mean_scores, times = load_scores(fname)
                all_scores[f"N{i}"] = mean_scores

            title = f"Decoding performance of {args.iter} iterations ({subject_id})"
            experiment = f"{subject_id}_N{start_n}_to_{end_n}"
            plot_scores(all_scores, times, title, OUTPUT_PATH, experiment)

        elif args.N and not args.drop_t:
            N = int(args.N)
            fname = get_subject_key(subject_files, subject_id, N, args.iter, f"_N{N}_")
            scores, mean_scores, times = load_scores(fname)
            all_scores = {f"Iter. : {i+1}" : score for i, score in enumerate(scores)}
            title = f"Decoding performance of {args.iter} iterations ({subject_id})"
            experiment = f"{subject_id}_N{N}"
            plot_scores(all_scores, times, title, OUTPUT_PATH, experiment, show_all=False)
       
        else:
            fname = get_subject_key(subject_files, subject_id, None, args.iter, f"")
            scores, mean_scores, times = load_scores(fname)
            all_scores = {f"Iter. : {i+1}" : score for i, score in enumerate(scores)}

            title = f"Decoding performance of {args.iter} iterations ({subject_id})"
            experiment = f"{subject_id}"
            plot_scores(all_scores, times, title, OUTPUT_PATH, experiment, show_all=False)

    # else:
    #     subject_files = eeg.get_subject_files(data_path=DATA_PATH, condition=condition + "**/", ext='json')
    #     experiment = condition.replace('/', '_').split('**')[0]

    #     pattern = f"_average_N{args.N}_repeated{args.iter}" if args.avg else f"_repeated{args.iter}"
    #     filenames = get_subjects_keys(subject_files, pattern)

    #     for fname in filenames:
    #         scores, mean_scores, times = load_scores(fname)
    #         subject_id = Path(fname).stem.split('_')[0]
    #         all_scores[subject_id] = mean_scores

    #     title = f"Decoding performance ({args.iter} iterations): all subjects"
    #     experiment = experiment + (f"N{args.N}_all_subjects" if not args.drop_t else "all_subjects")
    #     plot_scores(all_scores, times, title, OUTPUT_PATH, experiment, show_all=False)
    else:

        subject_files = eeg.get_subject_files(data_path=DATA_PATH, condition="**/S", ext='json')

        if args.N and "," in args.N:
            start_n, end_n = map(int, args.N.split(','))
            all_scores = {}

            for N in range(start_n, end_n):
                pattern = f"_average_N{N}_repeated{args.iter}" if args.avg else f"_N{N}_repeated{args.iter}"
                filenames = get_subjects_keys(subject_files, pattern)

                if not filenames:
                    logger.warning(f"No file found for N={N}")
                    continue

                all_subject_scores = []

                for fname in filenames:
                    scores, mean_scores, times = load_scores(fname)
                    all_subject_scores.append(np.mean(scores, axis=0))

                if all_subject_scores:
                    mean_over_subjects = np.mean(all_subject_scores, axis=0)
                    all_scores[f"N={N}"] = mean_over_subjects

            title = f"Decoding performances of all subjects, accross N ({args.iter} itér.)"
            experiment = f"mean_across_subjects_N_{start_n}_to_{end_n}"
            plot_scores(all_scores, times, title, OUTPUT_PATH, experiment, show_all=True)

        elif args.N is None:
            pattern = f"_repeated{args.iter}_decoding_results"
            # pattern = f"_average_N{args.N}_repeated{args.iter}" if args.avg else f"_N{args.N}_repeated{args.iter}"
            filenames = get_subjects_keys(subject_files, pattern)

            for fname in filenames:
                _, mean_scores, times = load_scores(fname)
                subject_id = Path(fname).stem.split('_')[0]
                all_scores[subject_id] = mean_scores

            title = f"Decoding performance ({50} iterations): all subjects/N_{1}"
            experiment = f"mean_across_subjects_no_avg"
            experiment += f"_N{args.N}" if not args.drop_t else ""
            plot_scores(all_scores, times, title, OUTPUT_PATH, experiment, show_all=False)