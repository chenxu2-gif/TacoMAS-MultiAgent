#!/bin/bash
#SBATCH --job-name=tacomas-demo
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

# --- Paths ----------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tacomas}"

# --- Instance range -------------------------------------------------
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1}"
SAMPLE_COUNT="${SAMPLE_COUNT:-}"
BATCH_SIZE="${BATCH_SIZE:-0}"
BATCH_SLEEP="${BATCH_SLEEP:-0}"
PER_INSTANCE_SLEEP="${PER_INSTANCE_SLEEP:-0}"

# --- TacoMAS hyperparameters (defaults match the paper) ------------
MAX_FAST_ROUNDS="${MAX_FAST_ROUNDS:-10}"   # round cap R
FAST_STEPS_PER_WINDOW="${FAST_STEPS_PER_WINDOW:-1}"
BD_CHECK_INTERVAL="${BD_CHECK_INTERVAL:-2}"  # slow update interval K
GRAPH_REWIRE_INTERVAL="${GRAPH_REWIRE_INTERVAL:-2}"
INIT_N_MIN="${INIT_N_MIN:-5}"  # |V_0| = 5 by default
INIT_N_MAX="${INIT_N_MAX:-5}"
POP_N_MIN="${POP_N_MIN:-2}"
POP_N_MAX="${POP_N_MAX:-20}"   # N_max
MAX_EDGE_EDITS="${MAX_EDGE_EDITS:-8}"   # B_E = 8
MIN_ITER_PER_AGENT="${MIN_ITER_PER_AGENT:-1}"
MAX_ITER_PER_AGENT="${MAX_ITER_PER_AGENT:-20}"
MAX_TOOL_CALLS_PER_SUBAGENT_ROUND="${MAX_TOOL_CALLS_PER_SUBAGENT_ROUND:-6}"
SKIP_META_INIT="${SKIP_META_INIT:-1}"

# --- Dataset (default: finance-benchmark; switch via DATASET_ID) ---
DATASET_ID="${DATASET_ID:-finance-benchmark}"
DATASET_LOCAL_PATH="${DATASET_LOCAL_PATH:-}"
DATASET_TEMPLATE_PATH="${DATASET_TEMPLATE_PATH:-}"
DATASET_ENV_NAME="${DATASET_ENV_NAME:-}"
DATASET_TOOLS="${DATASET_TOOLS:-}"

mkdir -p "${PROJECT_ROOT}/log" "${PROJECT_ROOT}/outputs"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MAX_TOOL_CALLS_PER_SUBAGENT_ROUND
export DISABLE_LLM_DETAIL_LOG="${DISABLE_LLM_DETAIL_LOG:-1}"
export DATASET_ID
[[ -n "${DATASET_LOCAL_PATH}" ]] && export DATASET_LOCAL_PATH
[[ -n "${DATASET_TEMPLATE_PATH}" ]] && export DATASET_TEMPLATE_PATH
[[ -n "${DATASET_ENV_NAME}" ]] && export DATASET_ENV_NAME
[[ -n "${DATASET_TOOLS}" ]] && export DATASET_TOOLS

# --- Models ---------------------------------------------------------
AGENT_MODEL="${AGENT_MODEL:-gemini/gemini-2.5-flash-lite}"
META_MODEL="${META_MODEL:-gemini/gemini-2.5-pro}"
EVAL_MODEL="${EVAL_MODEL:-openai/gpt-4o-mini}"
export EVAL_MODEL EVAL_API_BASE=""
export JUDGE_MODEL="${JUDGE_MODEL:-${EVAL_MODEL}}"

if [[ -n "${SAMPLE_COUNT}" ]]; then
  END_IDX="$((START_IDX + SAMPLE_COUNT))"
fi

COMMON_ARGS=(
  --max_fast_rounds "${MAX_FAST_ROUNDS}"
  --fast_steps_per_window "${FAST_STEPS_PER_WINDOW}"
  --bd_check_interval "${BD_CHECK_INTERVAL}"
  --graph_rewire_interval "${GRAPH_REWIRE_INTERVAL}"
  --init_n_min "${INIT_N_MIN}"
  --init_n_max "${INIT_N_MAX}"
  --pop_n_min "${POP_N_MIN}"
  --pop_n_max "${POP_N_MAX}"
  --max_edge_edits "${MAX_EDGE_EDITS}"
  --min_iterations_per_agent "${MIN_ITER_PER_AGENT}"
  --max_iterations_per_agent "${MAX_ITER_PER_AGENT}"
  --agent_model "${AGENT_MODEL}"
  --meta_model "${META_MODEL}"
)

if [[ "${SKIP_META_INIT}" == "1" ]]; then
  COMMON_ARGS+=(--skip_meta_init)
fi

if [[ "${BATCH_SIZE}" -gt 0 ]]; then
  echo "Running batched evolution: start_total=${START_IDX}, end_total=${END_IDX}, batch_size=${BATCH_SIZE}"
  python scripts/run_evolution.py \
    --start_total "${START_IDX}" \
    --end_total "${END_IDX}" \
    --batch_size "${BATCH_SIZE}" \
    --batch_sleep "${BATCH_SLEEP}" \
    --per_instance_sleep "${PER_INSTANCE_SLEEP}" \
    "${COMMON_ARGS[@]}"
else
  echo "Running single-range evolution: start=${START_IDX}, end=${END_IDX}"
  python scripts/run_evolution.py \
    --start "${START_IDX}" \
    --end "${END_IDX}" \
    --per_instance_sleep "${PER_INSTANCE_SLEEP}" \
    "${COMMON_ARGS[@]}"
fi
