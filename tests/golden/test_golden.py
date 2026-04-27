"""
Golden-prediction test.

Loads the frozen prediction from golden_prediction.json and asserts that the
current model produces exactly the same probability (within tolerance).

FAIL means: model behaviour changed unexpectedly — a retrain happened, a
preprocessing step was modified, or a dependency bumped changed numeric output.

To intentionally update the frozen value after a deliberate model change:
    python3 -c "
    import joblib, pandas as pd, json
    from pathlib import Path
    spec = json.loads(Path('tests/golden/golden_prediction.json').read_text())
    prob = float(joblib.load(spec['model_file']).predict_proba(
        pd.DataFrame([spec['customer']]))[:, 1][0])
    spec['churn_probability'] = round(prob, 6)
    Path('tests/golden/golden_prediction.json').write_text(json.dumps(spec, indent=2))
    print(f'Updated frozen probability: {prob:.6f}')
    "
"""

import json
from pathlib import Path

import pandas as pd
import pytest

GOLDEN_FILE = Path(__file__).parent / "golden_prediction.json"
MODEL_ROOT = Path(__file__).resolve().parent.parent.parent


def load_golden() -> dict:
    return json.loads(GOLDEN_FILE.read_text())


@pytest.mark.skipif(
    not GOLDEN_FILE.exists(),
    reason="golden_prediction.json not found",
)
def test_golden_prediction_unchanged():
    """
    Model must produce the same probability as the frozen value.
    A failure signals unintended model behaviour change.
    """
    try:
        import joblib
    except ImportError:
        pytest.skip("joblib not installed")

    spec = load_golden()
    model_path = MODEL_ROOT / spec["model_file"]

    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path} — run `make train` first")

    model = joblib.load(model_path)
    df = pd.DataFrame([spec["customer"]])
    predicted = float(model.predict_proba(df)[:, 1][0])
    frozen = spec["churn_probability"]
    tolerance = spec.get("tolerance", 1e-4)

    assert abs(predicted - frozen) <= tolerance, (
        f"Golden prediction shifted: expected {frozen:.6f}, "
        f"got {predicted:.6f} (diff={abs(predicted - frozen):.6f}, tol={tolerance}). "
        f"If this is intentional, update tests/golden/golden_prediction.json."
    )
