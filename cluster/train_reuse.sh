#!/bin/bash
#SBATCH --job-name=train_reuse_job
#SBATCH --output=/data/home/brahimhh/ter_project/logs/train_reuse_%A_S%a.out
#SBATCH --error=/data/home/brahimhh/ter_project/logs/train_reuse_%A_S%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

#SBATCH --array=2,3,5,7,8,9,10,11,13,14,16,17,18,19,22,23,25,26,27,28,29,31,32,33

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=haroun-hassan.brahim@etu.univ-amu.fr
#SBATCH --chdir=/data/home/brahimhh/ter_project/

/data/home/brahimhh/ter_project/reuse.sh $SLURM_ARRAY_TASK_ID
