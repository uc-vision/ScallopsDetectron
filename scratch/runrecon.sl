#!/bin/bash
#SBATCH --account=def-tkr25    # put your usercode for accounting
#SBATCH --gres=gpu:2              # Number of GPU(s), 1, 2 or 4
#SBATCH --cpus-per-task=6         # CPU cores/threads up to 64
#SBATCH --time=0-10:00            # wallclock time (DD-HH:MM) - up to 24 hours on our
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metashape
export agisoft_LICENSE=UCLICENSE4p@5053
python MetashapeReconstruction.py
