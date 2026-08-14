"""Extended measurement and analysis guides for ASI campaigns.

Complete documentation for executing and analyzing all measurement scenarios.
"""

# CAMPAIGN EXECUTION GUIDE - IPMNIST
IPMNIST_GUIDE = """
# IPMNIST Screening Campaign Guide

## Overview
IPMNIST tests learner robustness to visual distribution shift.
- Domain: Image classification with controlled shift
- Tasks: 200 sequential image classification tasks
- Metric: Per-task accuracy

## Arms to Test
- upgd_w_control: Baseline UPGD
- adamw_control: Adam baseline
- upgd_ema_norm: EMA normalizer
- All variants: norm_decay, step_size, weight_decay

## Execution
```bash
python measurement_cli.py ipmnist --arm upgd_w_control --n-tasks 200 --n-steps 5000 --seed 0
```

## Expected Performance
- Baseline: ~0.85 accuracy
- With normalization: ~0.87-0.88
- Top performers: ~0.89+

## Analysis
- Track accuracy over task sequence
- Identify when performance degrades
- Compare recovery rates across arms
"""

# CAMPAIGN EXECUTION GUIDE - SCR V2
SCR_GUIDE = """
# SCR v2 Validation Campaign Guide

## Overview
SCR v2 tests on slowly changing regression - synthetic but interpretable.
- Domain: Linear regression with concept drift
- Tasks: 100 sequential regression tasks
- Metric: MSE per task

## Arms to Test
- backprop_sgd_relu: SGD with ReLU
- adamw_baseline: Adam baseline
- upgd_w_baseline: UPGD baseline
- All variants: step size, weight decay

## Execution
```bash
python measurement_cli.py scr --arm upgd_w_baseline --n-tasks 100 --n-steps 1000 --seed 0
```

## Expected Performance
- Baseline MSE: ~0.01-0.02
- Optimized: ~0.005-0.01
- Best: ~0.001-0.005

## Analysis
- Plot MSE trajectory
- Identify learning curves
- Compare plasticity vs stability
"""

# CAMPAIGN EXECUTION GUIDE - EMNIST V3
EMNIST_GUIDE = """
# EMNIST v3 Validation Campaign Guide

## Overview
EMNIST tests protection mechanisms for handwritten digits.
- Domain: Extended MNIST with additional characters
- Tasks: 400 sequential digit classification
- Metric: Accuracy with protection

## Learners to Test
- upgd_w: Baseline
- adamw: Adam variant
- upgd_ema_norm_cbp: With CBP protection
- sgd_norm_cbp: SGD variant

## Execution
```bash
python measurement_cli.py emnist --learner upgd_ema_norm_cbp --n-tasks 400 --seed 0
```

## Expected Performance
- Without protection: ~0.90 accuracy
- With CBP: ~0.92-0.93
- Top: ~0.94+

## Analysis
- Measure protection effectiveness
- Compare across learner types
- Validate generalization
"""

# RESULT ANALYSIS GUIDE
ANALYSIS_GUIDE = """
# Result Analysis and Interpretation Guide

## 1. Loading Results
```python
from campaign_analysis import CampaignAnalyzer
import json

# Load campaign
with open('outputs/ipmnist_campaign/campaign_results.json') as f:
    results = json.load(f)

# Generate report
summary = CampaignAnalyzer.generate_summary_report(results)
```

## 2. Ranking Arms
```python
# Get ranked list
ranking = CampaignAnalyzer.rank_arms(results)

# Top 5
for arm, score in ranking[:5]:
    print(f"{arm}: {score:.4f}")
```

## 3. Cross-Campaign Comparison
```python
all_campaigns = {
    'ipmnist': ipmnist_results,
    'scr': scr_results,
    'emnist': emnist_results,
}

comparison = CampaignAnalyzer.cross_campaign_comparison(all_campaigns)
```

## 4. Statistical Validation
```python
from result_validation import ResultValidator

validator = ResultValidator()

# Significance test
sig = validator.significance_test(group1_results, group2_results)
print(f"p-value: {sig['p_value']:.6f}")

# Effect size
effect = validator.effect_size(group1_results, group2_results)
print(f"Cohen's d: {effect['cohens_d']:.4f}")
```

## 5. Visualization
```python
from result_validation import ResultVisualizer

viz = ResultVisualizer()

# Compare arms
viz.plot_arm_comparison(results_by_arm, output_path='comparison.png')

# Transfer curves
viz.plot_transfer_curve(source, target, output_path='transfer.png')
```

## 6. Publication Summary
```python
summary = generate_publication_summary('outputs/')

# Access key findings
for arm, score, campaign in summary['key_findings']['top_arms_across_campaigns']:
    print(f"{arm}: {score:.4f} ({campaign})")
```

## Interpretation Checklist
- [ ] All arms completed without errors
- [ ] Results show expected trends
- [ ] Statistical significance confirmed (p < 0.05)
- [ ] Effect sizes reasonable (|d| > 0.5)
- [ ] No outliers or data quality issues
- [ ] Consistency across seeds
- [ ] Transfer between domains validated
"""

# TROUBLESHOOTING GUIDE
TROUBLESHOOTING_GUIDE = """
# Troubleshooting Guide

## Issue: Smoke tests fail
**Cause**: Missing registration or import error
**Solution**: Run `python measurement_cli.py ipmnist --arm upgd_w_control` to test

## Issue: Campaign hangs
**Cause**: Infinite loop or deadlock
**Solution**: Check logs, increase timeout, restart

## Issue: Results show NaN
**Cause**: Numerical instability or divergence
**Solution**: Reduce step_size, increase weight_decay

## Issue: Low performance
**Cause**: Wrong hyperparameters or hyperparameter mismatch
**Solution**: Validate against preregistered values

## Issue: Results inconsistent across seeds
**Cause**: High variance or insufficient data
**Solution**: Run more seeds, check for randomness issues

## Issue: Memory error
**Cause**: Insufficient memory for task
**Solution**: Reduce batch size, simplify model, increase memory
"""

GUIDES = {
    "ipmnist": IPMNIST_GUIDE,
    "scr": SCR_GUIDE,
    "emnist": EMNIST_GUIDE,
    "analysis": ANALYSIS_GUIDE,
    "troubleshooting": TROUBLESHOOTING_GUIDE,
}

def print_guide(campaign_type: str):
    """Print guide for specific campaign type."""
    if campaign_type in GUIDES:
        print(GUIDES[campaign_type])
    else:
        print(f"Available guides: {list(GUIDES.keys())}")

def export_guides(output_dir: str = "docs"):
    """Export all guides to files."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for guide_name, guide_content in GUIDES.items():
        guide_file = output_dir / f"{guide_name}_guide.md"
        with open(guide_file, "w") as f:
            f.write(guide_content)
        print(f"✓ Exported {guide_file}")

if __name__ == "__main__":
    export_guides()
