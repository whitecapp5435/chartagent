#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 10:00:00
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
echo "Running python script"
python scripts/run_chartagent_misviz.py --manifest out/misviz_devval_manifest.jsonl --split dev,val --out out/misviz_devval_predictions.jsonl --model gpt-5-nano --metadata-model gpt-5-nano --max-steps 6 --max-side 1024 --runs-root out/chartagent_runs --cache-root out/chartagent_cache --resume --include-trace