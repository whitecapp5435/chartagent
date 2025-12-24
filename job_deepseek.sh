#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 23:00:00
#SBATCH --mem=64G
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:2
#SBATCH -A grp_vgupt140
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"

echo "Loading python 3 from anaconda module"
module load mamba/latest
echo "Loading python environment"
source activate pj2
echo "Running python script"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHARTAGENT_MAX_NEW_TOKENS=256
export CHARTAGENT_DEEPSEEK_DEVICE_MAP=auto
export CHARTAGENT_DEEPSEEK_USE_CACHE=0
export CHARTAGENT_DEEPSEEK_MAX_CANDIDATE_SIDE=384
nvidia-smi
python scripts/run_chartagent_misviz.py --manifest out/misviz_devval_manifest.jsonl --split dev,val --out out/misviz_devval_predictions_deepseekvl2.jsonl --model deepseek-ai/deepseek-vl2-tiny --metadata-model gpt-5-nano --max-steps 6 --max-attached-images 1 --max-side 384 --runs-root out/chartagent_runs --cache-root out/chartagent_cache --resume --include-trace
