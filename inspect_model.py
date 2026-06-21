import joblib
from pathlib import Path

# adjust this path if running from a different location
model_path = Path("models/model.joblib")

model = joblib.load(model_path)

print("Type:", type(model))
print("Module:", type(model).__module__)
print("Class name:", type(model).__name__)

# If it's a composite, show what's inside
if hasattr(model, "estimators_"):
    print("\nHas estimators_ (ensemble/stacking):")
    for est in model.estimators_:
        print("  -", type(est))

if hasattr(model, "steps"):
    print("\nHas steps (Pipeline):")
    for name, step in model.steps:
        print(f"  - {name}: {type(step)}")

if hasattr(model, "named_estimators_"):
    print("\nHas named_estimators_ (StackingRegressor/VotingRegressor):")
    for name, est in model.named_estimators_.items():
        print(f"  - {name}: {type(est)}")

if hasattr(model, "regressor_"):
    print("\nTransformedTargetRegressor fitted inner regressor (regressor_):")
    print("  -", type(model.regressor_))
    inner = model.regressor_
    if hasattr(inner, "steps"):
        print("    Pipeline steps:")
        for name, step in inner.steps:
            print(f"      - {name}: {type(step)}")
    if hasattr(inner, "named_estimators_"):
        print("    named_estimators_:")
        for name, est in inner.named_estimators_.items():
            print(f"      - {name}: {type(est)}")
elif hasattr(model, "regressor"):
    print("\nTransformedTargetRegressor unfitted inner regressor (regressor):")
    print("  -", type(model.regressor))

if hasattr(model, "transformer_"):
    print("\nTarget transformer_:", type(model.transformer_))
elif hasattr(model, "transformer"):
    print("\nTarget transformer:", type(model.transformer))
elif hasattr(model, "func"):
    print("\nTarget func:", model.func)
    print("Target inverse_func:", model.inverse_func)