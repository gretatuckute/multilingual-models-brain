#!/bin/bash
# Fit the five causal-text context runs after extraction, using the same
# 100-second held-out-story scoring trim as the final AuriStream analysis.

#SBATCH --job-name=ns_text_fit
#SBATCH --output=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_fit_%A_%a.out
#SBATCH --error=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_fit_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=sapphire
#SBATCH --mail-type=FAIL

set -euo pipefail

REPRO_ROOT=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction
PYTHON=/n/holylabs/kempner_gtuckute_lab/Lab/envs/auristream-alpha/bin/python

MODEL_KEYS=(
    gpt2_xl
    gpt_j_6b
    gpt_j_6b
    qwen3_8b
    qwen3_8b
)
SOURCE_NAMES=(
    gpt2_xl_context20p48s_3shift
    gpt_j_6b_context100words_3shift
    gpt_j_6b_context20p48s_3shift
    qwen3_8b_context100words_3shift
    qwen3_8b_context20p48s_3shift
)
OUTPUT_NAMES=(
    gpt2_xl_context20p48s_ramptrim100s_alllayers_3shift
    gpt_j_6b_context100words_ramptrim100s_alllayers_3shift
    gpt_j_6b_context20p48s_ramptrim100s_alllayers_3shift
    qwen3_8b_context100words_ramptrim100s_alllayers_3shift
    qwen3_8b_context20p48s_ramptrim100s_alllayers_3shift
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script as array 0-4}
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_KEYS[@]} )); then
    echo "Array index ${TASK_ID} is outside the five-run grid" >&2
    exit 2
fi

MODEL_KEY=${MODEL_KEYS[$TASK_ID]}
SOURCE_RUN=${REPRO_ROOT}/runs/${SOURCE_NAMES[$TASK_ID]}
OUTPUT_DIR=${REPRO_ROOT}/runs/${OUTPUT_NAMES[$TASK_ID]}

mkdir -p "${REPRO_ROOT}/runs/logs"

exec "${PYTHON}" "${REPRO_ROOT}/fit_text_ramp_trim.py" \
    --model-key "${MODEL_KEY}" \
    --source-run "${SOURCE_RUN}" \
    --output-dir "${OUTPUT_DIR}" \
    --test-trim-start-bins 50
