import setup
from train_func import *
from utils import get_file_name
from configs.config import *
# import faulthandler
# faulthandler.enable()


if __name__ == "__main__":
    # SETTING ARGUMENT PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, default=CLEAN_SUBJECTS_PATH,
                        help='Dataset path')

    parser.add_argument('--output_dir', type=str, required=True, default=OUTPUT_PATH,
                        help='Output path for results.')

    parser.add_argument('--reuse', type=bool, required=False, default=False,
                        help='Averaging technique with reuse.')

    parser.add_argument('--factor', type=float, required=False, default=None,
                        help='The foctor of augmentation when averaging with reuse technic.')

    parser.add_argument('--start_n', type=int, required=True, default=2,
                        help='Mini-average start value to loop over.')

    parser.add_argument('--end_n', type=int, required=True, default=11,
                        help='Mini-average end value to loop over.')

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
    start = args.start_n
    end = args.end_n
    reuse = True if args.reuse else False
    factor = args.factor if args.factor else None
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

    n_jobs = 4 #min(FOLD_ITER, get_n_jobs())
    logger.info(f"Number of cpus to run iterations: {n_jobs} ")

    # DATA PROCESSING AND LABELIZATION
    subject_id = f"S{int(args.subject):02d}"
    
    filename = get_file_name(subject_id, subject_files)

    experiment = f"dropped{'_trials_' if drop_trials else '_'}volume_fussy/micro_avg/"
    if reuse:
        logger.info("Lauching micro average with reuse")
        experiment += f"reuse_{FOLD_ITER}/"
    else:
        logger.info("Lauching micro average without reuse")
        experiment += f"not_reuse_{FOLD_ITER}/"
    
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
    experiment += subject["fname"]
    output_dir = OUTPUT_PATH / f"train/{experiment}"

    fname = subject['fname']

    for N in tqdm(range(args.start_n, args.end_n), colour="ORANGE", desc="Micro-average...:"):
        new_fname = fname + f"_average_N{N}"
        new_fname += f'_f{factor}' if factor else ''

        start = datetime.now()
        all_scores = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(train)(
            n_splits=n_splits,
            subject=subject,
            random_state=idx,
            average=True,
            times=TIMES, 
            fname=new_fname,
            N=N,
            reuse=reuse,
            factor=factor,
            model_path=None, # MODEL_PATH / "train/",
            output_dir=output_dir,
            save_json=True,
            save_image=True,
            ) for idx in range(FOLD_ITER))
        
        end = datetime.now()
        elapsed = (end - start).total_seconds()

        logger.info(f"Training duration of subject: {subject_id}, N: {N}, iterations: {FOLD_ITER} -> {elapsed:.4f} sec.")
        logger.info(f"Started at: {start} and ended at: {end}")

        all_scores = np.array(all_scores)
        
        new_fname += f'_repeated{FOLD_ITER}'

        plot_fold_loop_history(
            scores=all_scores, 
            times=TIMES, 
            fname=new_fname, 
            output_dir=output_dir, 
            plot=False
            )

        save_results(
            scores=all_scores,
            times=TIMES,
            fname=new_fname,
            output_dir=output_dir,
            title='iteration',
            )

