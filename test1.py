"""
Run this locally to verify skops can serialize the model with the
trusted_types list before pushing to CI. This avoids burning CI runs
on serialization issues.

Usage:
    python test_skops_save.py
"""
import joblib
from pathlib import Path
import skops.io as sio

model_path = Path("models/model.joblib")
model = joblib.load(model_path)

trusted_types = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "sklearn.utils._bunch.Bunch",
]

print("Attempting skops dump...")
out_path = "test_model.skops"
sio.dump(model, out_path)
print("Dump succeeded:", out_path)

print("\nAttempting skops load with trusted_types...")
try:
    loaded = sio.load(out_path, trusted=trusted_types)
    print("Load succeeded with current trusted_types list!")
except Exception as e:
    print("Load FAILED. Full error below:\n")
    print(e)
    print("\nTrying to get the exact list of untrusted types skops requires...")
    try:
        unknown = sio.get_untrusted_types(file=out_path)
        print("Untrusted types found:", unknown)
        print("\nUpdate trusted_types in evaluation.py to include all of these.")
    except Exception as inner_e:
        print("Could not auto-detect untrusted types:", inner_e)

# cleanup
Path(out_path).unlink(missing_ok=True)