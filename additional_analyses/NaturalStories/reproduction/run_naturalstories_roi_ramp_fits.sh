#!/bin/bash
# Fit the five released Natural Stories language fROIs for all trained
# AuriStream horizon models, using the onset-ramp-corrected held-out score.
#
# Submit the complete 8-model x 5-fROI grid with bounded concurrency:
#   sbatch --array=0-39%8 run_naturalstories_roi_ramp_fits.sh

#SBATCH --job-name=ns_roi_ramp
#SBATCH --output=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/AuriStream-alpha/evals_neural/out/ns_roi_ramp_%A_%a.out
#SBATCH --error=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/AuriStream-alpha/evals_neural/err/ns_roi_ramp_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --partition=sapphire
#SBATCH --mail-type=FAIL

set -euo pipefail

REPRO_ROOT=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/reproduction
RESPONSE_ROOT=/n/holylabs/kempner_gtuckute_lab/Lab/code_repos/multilingual-models-brain-auristream/additional_analyses/NaturalStories/response
PYTHON=/n/holylabs/kempner_gtuckute_lab/Lab/envs/auristream-alpha/bin/python

MODELS=(
    TuKoResearch/AuriStream100M_1Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream100M_10Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream100M_20Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream100M_40Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream100M_60Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream100M_80Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream7BDeep_1Pred_BigAudioDataset_500k
    TuKoResearch/AuriStream7BDeep_40Pred_BigAudioDataset_500k
)
ROIS=(AntTemp IFG IFGorb MFG PostTemp)
LAYERS_100M=(0 1 2 3 4 5 6 7 8 9 10 11 12)
LAYERS_7B=(0 3 6 9 12 15 18 21 24 27 30 33 36 39 42 45 48 51 54 57 60 63 66 69 72 75 78 81 84 87 90 93 96)

TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}
MODEL_INDEX=$((TASK_ID / ${#ROIS[@]}))
ROI_INDEX=$((TASK_ID % ${#ROIS[@]}))
if (( MODEL_INDEX >= ${#MODELS[@]} )); then
    echo "Array index ${TASK_ID} is outside the 8-model x 5-fROI grid" >&2
    exit 2
fi

MODEL=${MODELS[$MODEL_INDEX]}
ROI=${ROIS[$ROI_INDEX]}
MODEL_SHORT=${MODEL#TuKoResearch/}
if [[ ${MODEL_SHORT} == AuriStream100M_* ]]; then
    LAYER_TAG=alllayers
    LAYERS=("${LAYERS_100M[@]}")
else
    LAYER_TAG=every3
    LAYERS=("${LAYERS_7B[@]}")
fi

OUTPUT_DIR=${REPRO_ROOT}/runs/${MODEL_SHORT}_roi-${ROI}_ramptrim100s_${LAYER_TAG}_3shift
RESPONSE_FILE=${RESPONSE_ROOT}/d_shift_3_${ROI}

cd "${REPRO_ROOT}"
exec "${PYTHON}" fit_auristream_ramp_trim.py \
    --source_model "${MODEL}" \
    --response_file "${RESPONSE_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --layers "${LAYERS[@]}"
