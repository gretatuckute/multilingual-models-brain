#!/bin/bash
# Extract causal text-model Natural Stories representations for the two
# prespecified context definitions. Set SMOKE=true to extract story 1 only.

#SBATCH --job-name=ns_text_extract
#SBATCH --output=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_extract_%A_%a.out
#SBATCH --error=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction/runs/logs/ns_text_extract_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_gtuckute_lab
#SBATCH --gres=gpu:1
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
CONTEXT_MODES=(
    20.48seconds
    100words
    20.48seconds
    100words
    20.48seconds
)
RUN_NAMES=(
    gpt2_xl_context20p48s_3shift
    gpt_j_6b_context100words_3shift
    gpt_j_6b_context20p48s_3shift
    qwen3_8b_context100words_3shift
    qwen3_8b_context20p48s_3shift
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script as array 0-4}
if (( TASK_ID < 0 || TASK_ID >= ${#MODEL_KEYS[@]} )); then
    echo "Array index ${TASK_ID} is outside the five-run grid" >&2
    exit 2
fi

MODEL_KEY=${MODEL_KEYS[$TASK_ID]}
CONTEXT_MODE=${CONTEXT_MODES[$TASK_ID]}
OUTPUT_DIR=${REPRO_ROOT}/runs/${RUN_NAMES[$TASK_ID]}
SMOKE=${SMOKE:-false}
STORY_ARGS=()
if [[ ${SMOKE} == true ]]; then
    STORY_ARGS=(--stories 1)
fi

export TOKENIZERS_PARALLELISM=false
mkdir -p "${REPRO_ROOT}/runs/logs"

exec "${PYTHON}" "${REPRO_ROOT}/reproduce.py" \
    --model-key "${MODEL_KEY}" \
    --causal-context-mode "${CONTEXT_MODE}" \
    --causal-context-seconds 20.48 \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --extract-only \
    "${STORY_ARGS[@]}"
