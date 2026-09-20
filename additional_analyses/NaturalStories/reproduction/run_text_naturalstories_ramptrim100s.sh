#!/bin/bash
# Fit an already-extracted text model on Natural Stories, omitting the first
# 100 seconds only from held-out-story correlation scoring.

#SBATCH --job-name=ns_text_ramp
#SBATCH --output=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_ramp_%j.out
#SBATCH --error=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_ramp_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --partition=sapphire
#SBATCH --mail-type=FAIL

set -euo pipefail

REPRO_ROOT=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction
PYTHON=/n/holylabs/kempner_gtuckute_lab/Lab/envs/auristream-alpha/bin/python
MODEL_KEY=${MODEL_KEY:-gpt2_xl}
SOURCE_RUN=${SOURCE_RUN:-${REPRO_ROOT}/runs/${MODEL_KEY}_3shift}
OUTPUT_DIR=${OUTPUT_DIR:-${REPRO_ROOT}/runs/${MODEL_KEY}_ramptrim100s_alllayers_3shift}

mkdir -p "${REPRO_ROOT}/runs/logs"

exec "${PYTHON}" "${REPRO_ROOT}/fit_text_ramp_trim.py" \
    --model-key "${MODEL_KEY}" \
    --source-run "${SOURCE_RUN}" \
    --output-dir "${OUTPUT_DIR}" \
    --test-trim-start-bins 50
