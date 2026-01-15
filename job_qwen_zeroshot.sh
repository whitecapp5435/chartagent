#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 23:00:00
#SBATCH --mem=32G
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -A grp_vgupt140
#SBATCH --mail-type=ALL
#SBATCH --mail-user="%u@asu.edu"

echo "Loading python 3 from anaconda module"
module load mamba/latest
echo "Loading python environment"
source activate pj2-qwen
echo "Running zeroshot baseline"

# Optional knobs:
export CHARTAGENT_MAX_NEW_TOKENS=256
# export QWEN3_VL_ATTN_IMPL=flash_attention_2

nvidia-smi

python scripts/run_zeroshot_misviz.py \
  --manifest out/misviz_devval_manifest.jsonl \
  --split dev,val \
  --out out/misviz_devval_predictions_qwen3vl_zeroshot.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --max-side 1024 \
  --resume

