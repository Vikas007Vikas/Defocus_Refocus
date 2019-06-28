#!/bin/bash
#BATCH -A $USER
#SBATCH -n 10
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module add cuda/8.0
module add cudnn/7-cuda-8.0
python main.py --train_Sharp_path ./output_new/train/sharp/ --train_focus_path ./output_new/train/focus/ --train_Blur_path ./output_new/train/blur --test_with_train True --save_test_result True
