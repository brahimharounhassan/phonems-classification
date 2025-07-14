import setup
from train_func import *
from utils import get_file_name
from configs.config import *
import os

if __name__ == "__main__":
    # SETTING ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, default=CLEAN_SUBJECTS_PATH,
                        help='Dataset path')

    parser.add_argument('--output_dir', type=str, required=True, default=OUTPUT_PATH,
                        help='Output path for results.')

    parser.add_argument('--include', type=str, required=False, default=None,
                        help='Drop trials or not.')

    parser.add_argument('--n_splits', type=int, required=False, default=5,
                        help='Number of K-fold splits.')    
    
    parser.add_argument('--iter', type=int, required=False, default=10,
                        help='Cross-validation repetition number.')
    
    parser.add_argument('--subject', type=str, required=True, default=3,
                        help='Subject num.')
    
    args = parser.parse_args()

    # SETTING UP CLASSES INSTANCIATION
    CONFIG_FILE = "configs/config.yml"

    eeg = EEG(
        n_channels=N_CHANNELS,
        sfreq=SFREQ
    )

    CLEAN_SUBJECTS_PATH = Path(args.data_dir)
    OUTPUT_PATH = Path(args.output_dir)
    FOLD_ITER = args.iter
    include = np.array(args.include.split(',')).astype(
        int) if args.include else []
    
    drop_trials = True if len(include) > 0 else False
    
    # DATA READING
    subject_files = eeg.get_subject_files(
        data_path=CLEAN_SUBJECTS_PATH,
        condition='/**/',
        ext=FILE_EXT
    )

    # MODEL BUILDING
    n_splits = args.n_splits

    n_cpus = get_n_jobs()
    n_jobs = min(FOLD_ITER, n_cpus)
    logger.info(f"Number of cpus to run iterations: {n_jobs} ")

    subject_id = f"S{int(args.subject):02d}"
    filename = get_file_name(subject_id, subject_files)
    
    # DATA PROCESSING AND LABELIZATION
    # for filename in tqdm(subject_files, colour="BLUE", desc="Reading subject data...:"):
    text_extension = f"/include_{'_'.join(map(str, include))}/" if len(include) > 0 else ''
    experiment = f"dropped{'_trials_' if drop_trials else '_'}volume_fussy/no_avg_{FOLD_ITER}/" + text_extension
    

    subject = eeg.get_subject(
        filename=filename,
        time_points=N_TIMES,
        sfreq=SFREQ
    )

    TIMES = subject["t"].flatten()

    subject = eeg.drop_trials(
        subject=subject,
        drop_trials=drop_trials,
        include=include,
        drop_volume=True,
        drop_fussy=True,
    )

    subject = eeg.encode_labels(subject)

    # MODEL TRAINING
    fname = subject["fname"]
    experiment += fname
    output_dir = OUTPUT_PATH / f"train/{experiment}"
    
    start = datetime.now()
    iter_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(train)(
        n_splits=n_splits,
        subject=subject,
        random_state=idx,
        average=False,
        times=TIMES, 
        fname=fname,
        N=None,
        reuse=False,
        factor=None,
        model_path=None, # MODEL_PATH / "train/",
        output_dir=output_dir,
        save_json=False,
        save_image=True,
        ) for idx in tqdm(range(FOLD_ITER), colour='WHITE', desc='Training...' ))
    
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    logger.info(f"Training duration of subject: {subject_id}, iterations: {FOLD_ITER} -> {elapsed:.4f} sec.")
    logger.info(f"Started at: {start} and ended at: {end}")
    
    fname = fname+f'_repeated{FOLD_ITER}'

    iter_scores = np.array(iter_scores)

    plot_fold_loop_history(
        scores=iter_scores, 
        times=TIMES, 
        fname=fname, 
        output_dir=output_dir, 
        plot=False
        )
    
    save_results(
        scores=iter_scores,
        times=TIMES,
        fname=fname,
        output_dir=output_dir,
        title='iteration',
        )    
    

