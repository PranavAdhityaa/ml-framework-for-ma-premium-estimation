"""
run_pipeline.py
================
Master script: runs the entire project from data generation to SHAP analysis.
Execute this to reproduce all results from scratch.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("Data Generation",      "src/data/generate_data.py"),
    ("Data Cleaning",        "src/data/clean_data.py"),
    ("Feature Engineering",  "src/features/feature_engineering.py"),
    ("EDA",                  "src/eda.py"),
    ("Model Training",       "src/models/train_models.py"),
    ("SHAP Analysis",        "src/models/shap_analysis.py"),
]

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ FAILED: {name} (exit code {result.returncode})")
        sys.exit(1)
    print(f"\n✓ {name} completed in {elapsed:.1f}s")

if __name__ == "__main__":
    print("=" * 60)
    print("  M&A Acquisition Premium Predictor — Full Pipeline")
    print("=" * 60)
    total_start = time.time()
    for name, script in STEPS:
        run_step(name, script)
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE in {total:.1f}s")
    print(f"{'='*60}")
    print("\nOutputs:")
    print("  data/processed/ma_deals_features.csv  — final dataset")
    print("  results/model_comparison.csv           — all model metrics")
    print("  results/figures/                       — 13 figures")
    print("  results/shap/finance_insights.csv      — quantified findings")
    print("  reports/final_report.md                — full report")


def run_improvements():
    """Run all three improvement modules."""
    improvement_steps = [
        ("Real Data Validation",        "src/validation/real_data_validation.py"),
        ("Optuna HPO",                   "src/models/optuna_tuning.py"),
        ("API Test Suite",               "src/api/test_api.py"),
    ]
    print("\n" + "="*60)
    print("  RUNNING IMPROVEMENT MODULES")
    print("="*60)
    for name, script in improvement_steps:
        run_step(name, script)

if __name__ == "__main__" and "--with-improvements" in __import__("sys").argv:
    for name, script in STEPS:
        run_step(name, script)
    run_improvements()
