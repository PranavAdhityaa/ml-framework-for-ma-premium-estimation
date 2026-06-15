import os, re

files_to_fix = [
    "src/data/generate_data.py",
    "src/data/clean_data.py",
    "src/data/collect_data.py",
    "src/features/feature_engineering.py",
    "src/models/train_models.py",
    "src/models/shap_analysis.py",
    "src/models/optuna_tuning.py",
    "src/validation/real_data_validation.py",
    "src/api/app.py",
    "src/api/test_api.py",
    "src/eda.py",
]

# Get the absolute path of wherever this script is sitting
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Convert to forward slashes so Python is happy on Windows too
PROJECT_ROOT_FWD = PROJECT_ROOT.replace("\\", "/")

OLD_PATH = "/home/claude/ma_premium_predictor"

fixed = []
for rel_path in files_to_fix:
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"  SKIP (not found): {rel_path}")
        continue
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    if OLD_PATH in content:
        new_content = content.replace(OLD_PATH, PROJECT_ROOT_FWD)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        fixed.append(rel_path)
        print(f"  FIXED: {rel_path}")
    else:
        print(f"  OK (no change needed): {rel_path}")

print(f"\nDone. Fixed {len(fixed)} files.")
print(f"Project root set to: {PROJECT_ROOT_FWD}")