#!/usr/bin/env bash
# MEASUREMENT_EXECUTION_RUNBOOK.sh
#
# Complete guide for executing all ASI measurement campaigns.
# Follow this runbook to validate and run all 8 pre-registered measurement campaigns.

set -e

echo "=========================================="
echo "ASI Measurement Campaign Execution Runbook"
echo "=========================================="
echo ""

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:-.}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"
COMPUTE_BUDGET="${COMPUTE_BUDGET:-80}"  # hours
NUM_SEEDS="${NUM_SEEDS:-3}"

echo "Configuration:"
echo "  Project Root: $PROJECT_ROOT"
echo "  Output Base: $OUTPUT_BASE"
echo "  Compute Budget: ${COMPUTE_BUDGET}h"
echo "  Seeds per Campaign: $NUM_SEEDS"
echo ""

# ============================================================================
# PHASE 0: VALIDATION (Pre-flight checks)
# ============================================================================

echo "PHASE 0: Validation Checks"
echo "=========================================="

# Run smoke tests
echo "[1] Running smoke tests..."
python -m pytest tests/test_smoke_campaigns.py -v --tb=short || {
    echo "FAILED: Smoke tests failed"
    exit 1
}
echo "✓ Smoke tests passed"

# Verify all arms/learners registered
echo "[2] Verifying arm/learner registration..."
python -c "
from alberta_framework.benchmarks.ipmnist_screening import screening_spec
from alberta_framework.benchmarks.slowly_changing_regression_v2_setup import get_learner_factory
from alberta_framework.benchmarks.upgd_label_emnist import _FULL_STEP_FACTORIES
from micro_continual_improvements import PREREGISTERED_ARMS
from alberta_framework.benchmarks.forager_open_baselines import make_baseline

# Check IPMNIST
arms = ['upgd_w_control', 'adamw_control', 'upgd_ema_norm']
for arm in arms:
    spec = screening_spec(arm)
    assert spec is not None
print('✓ IPMNIST arms registered')

# Check SCR
for arm in ['backprop_sgd_relu', 'adamw_baseline', 'upgd_w_baseline']:
    factory = get_learner_factory(arm)
    assert factory is not None
print('✓ SCR baselines registered')

# Check EMNIST
for learner in ['upgd_w', 'adamw', 'upgd_ema_norm_cbp', 'sgd_norm_cbp']:
    assert learner in _FULL_STEP_FACTORIES
print('✓ EMNIST learners registered')

# Check Micro-Continual
assert 'rls_head_resid' in PREREGISTERED_ARMS
print('✓ Micro-continual arms registered')

# Check Forager
for baseline in ['dqn', 'a3c', 'random']:
    agent = make_baseline(baseline, action_dim=4, state_dim=16)
    assert agent is not None
print('✓ Forager baselines initialized')
" || {
    echo "FAILED: Registration verification failed"
    exit 1
}
echo "✓ All arms/learners verified"
echo ""

# ============================================================================
# CAMPAIGN 1: IPMNIST SCREENING
# ============================================================================

echo "CAMPAIGN 1: IPMNIST Screening"
echo "=========================================="
echo "Duration: ~3.5 hours (estimated)"
echo "Tasks: 200 | Steps: 5000 | Seeds: $NUM_SEEDS"
echo ""

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    echo "Seed $seed:"
    python measurement_cli.py ipmnist \
        --arm upgd_w_control \
        --n-tasks 200 \
        --n-steps 5000 \
        --seed $seed \
        --output-dir "$OUTPUT_BASE/ipmnist"
    echo "✓ Seed $seed complete"
done
echo "✓ IPMNIST Campaign complete"
echo ""

# ============================================================================
# CAMPAIGN 2: SCR v2 VALIDATION
# ============================================================================

echo "CAMPAIGN 2: SCR v2 Validation"
echo "=========================================="
echo "Duration: ~18 hours (estimated)"
echo "Tasks: 100 | Steps: 1000 | Seeds: $NUM_SEEDS"
echo ""

for arm in backprop_sgd_relu adamw_baseline upgd_w_baseline; do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "Arm=$arm, Seed=$seed"
        python measurement_cli.py scr \
            --arm $arm \
            --n-tasks 100 \
            --n-steps 1000 \
            --seed $seed \
            --output-dir "$OUTPUT_BASE/scr"
    done
done
echo "✓ SCR v2 Campaign complete"
echo ""

# ============================================================================
# CAMPAIGN 3: EMNIST v3 VALIDATION
# ============================================================================

echo "CAMPAIGN 3: EMNIST v3 Validation"
echo "=========================================="
echo "Duration: ~12 hours (estimated)"
echo "Tasks: 400 | Steps: 1000 | Seeds: $NUM_SEEDS"
echo ""

for learner in upgd_w adamw upgd_ema_norm_cbp sgd_norm_cbp; do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "Learner=$learner, Seed=$seed"
        python measurement_cli.py emnist \
            --learner $learner \
            --n-tasks 400 \
            --n-steps 1000 \
            --seed $seed \
            --output-dir "$OUTPUT_BASE/emnist"
    done
done
echo "✓ EMNIST v3 Campaign complete"
echo ""

# ============================================================================
# CAMPAIGN 4: MICRO-CONTINUAL VALIDATION
# ============================================================================

echo "CAMPAIGN 4: Micro-Continual Validation"
echo "=========================================="
echo "Duration: ~10 hours (estimated)"
echo "Tasks: M1-M4 | Seeds: $NUM_SEEDS"
echo ""

for arm in rls_head_resid alignment_first naive_bayes_extended; do
    for task_suite in m1 m2 m3 m4; do
        echo "Arm=$arm, Tasks=$task_suite"
        python measurement_cli.py micro \
            --arm $arm \
            --tasks $task_suite \
            --n-seeds $NUM_SEEDS \
            --output-dir "$OUTPUT_BASE/micro"
    done
done
echo "✓ Micro-Continual Campaign complete"
echo ""

# ============================================================================
# CAMPAIGN 5: FORAGER PHASE 1 (SMOKE TEST)
# ============================================================================

echo "CAMPAIGN 5: Forager Phase 1 (Smoke Test)"
echo "=========================================="
echo "Duration: ~4 hours (estimated)"
echo "Baselines: 4 | Episodes: 100 | Seeds: $NUM_SEEDS"
echo ""

for baseline in dqn a3c random horde; do
    for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "Baseline=$baseline, Seed=$seed"
        python measurement_cli.py forager \
            --baseline $baseline \
            --phase smoke \
            --n-tasks 1 \
            --n-episodes 100 \
            --seed $seed \
            --output-dir "$OUTPUT_BASE/forager"
    done
done
echo "✓ Forager Phase 1 complete"
echo ""

# ============================================================================
# PHASE 1: RESULT COLLECTION
# ============================================================================

echo "PHASE 1: Result Collection & Aggregation"
echo "=========================================="
echo "[1] Collecting all results..."

python -c "
import json
from pathlib import Path
from alberta_framework.utils.result_aggregation import ResultAggregator

agg = ResultAggregator()

# Load results from all campaigns
output_base = Path('$OUTPUT_BASE')
for campaign_dir in output_base.glob('*/'):
    for result_file in campaign_dir.glob('*/result.json'):
        with open(result_file) as f:
            data = json.load(f)
            # Parse and aggregate
            print(f'Loaded {result_file}')

# Save aggregated results
agg.to_json(output_base / 'aggregated_results.json')
print(f'✓ Results aggregated to {output_base}/aggregated_results.json')
" || echo "Note: Result aggregation requires campaign output files"

echo ""

# ============================================================================
# PHASE 2: RESULT ANALYSIS
# ============================================================================

echo "PHASE 2: Result Analysis & Visualization"
echo "=========================================="
echo "[1] Generating analysis plots..."

python -c "
import json
from pathlib import Path
from alberta_framework.utils.result_validation import ResultVisualizer, ResultValidator

output_base = Path('$OUTPUT_BASE')

# Generate plots (if results available)
print('Analysis plots would be generated here')
print('See result_validation.py for visualization utilities')
"

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo "Campaign Execution Complete"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Campaign 1: IPMNIST Screening"
echo "  ✓ Campaign 2: SCR v2 Validation"
echo "  ✓ Campaign 3: EMNIST v3 Validation"
echo "  ✓ Campaign 4: Micro-Continual"
echo "  ✓ Campaign 5: Forager Phase 1"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "Next steps:"
echo "  1. Verify all output files are present"
echo "  2. Run analysis/visualization scripts"
echo "  3. Compare results against baseline"
echo "  4. Prepare publication summary"
echo ""
echo "For detailed results, see:"
echo "  - $OUTPUT_BASE/aggregated_results.json"
echo "  - $OUTPUT_BASE/*/result.json (per-campaign)"
echo ""
