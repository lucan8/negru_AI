import pickle
from pathlib import Path
from typing import Any
from io import StringIO

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

from solution import FEATURES, recursive_forecast_for_index

app = Flask(__name__)

DATA_PATH = Path("data/est_hourly.parquet")
MODELS_ROOT = Path("xgboost") / "v15"
EXCLUDED_TARGETS = {"NI", "PJM_Load"}

CACHE: dict[str, Any] = {
    "df": None,
    "models": None,
    "included_targets": None,
    "skipped_targets": None,
}

def _pick_model_file_for_target(target: str):
    model_dir = MODELS_ROOT / target / "models"
    if not model_dir.exists():
        return None

    candidates = [p for p in model_dir.rglob("*") if p.is_file()]
    if not candidates:
        return None

    # Prefer a direct file named "model" if present; otherwise latest modified.
    for cand in candidates:
        if cand.name == "model":
            return cand
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_context():
    if CACHE["df"] is not None:
        return

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    models = {}
    included_targets = []
    skipped_targets = {}

    for target in df.columns:
        if target in EXCLUDED_TARGETS:
            skipped_targets[target] = "excluded_by_config"
            continue

        model_path = _pick_model_file_for_target(target)
        if model_path is None:
            skipped_targets[target] = "model_not_found"
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        models[target] = model
        included_targets.append(target)

    CACHE["df"] = df
    CACHE["models"] = models
    CACHE["included_targets"] = included_targets
    CACHE["skipped_targets"] = skipped_targets


def _parse_input_date(value: str | None):
    if value is None:
        raise ValueError("Missing 'date'. Provide YYYY-MM-DD.")
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Invalid 'date'. Expected format YYYY-MM-DD.")
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _predict_aggregate_for_day(day_start: pd.Timestamp):
    _load_context()

    df = CACHE["df"]
    models = CACHE["models"]
    included_targets = CACHE["included_targets"]
    skipped_targets = dict(CACHE["skipped_targets"])

    day_end = day_start + pd.Timedelta(days=1)
    day_hours = pd.date_range(start=day_start, end=day_end, freq="h", inclusive="left")
    aggregate = np.zeros(len(day_hours), dtype=float)
    used_targets = []

    for target in included_targets:
        history = df[target].dropna().sort_index()
        if len(history) < 168:
            skipped_targets[target] = "insufficient_history"
            continue

        history_last = history.index.max()
        print(f"{target}: {history_last}")
        forecast_start = history_last + pd.Timedelta(hours=1)

        # Predict up to the requested day end, then slice just the requested day.
        if day_end <= forecast_start:
            skipped_targets[target] = "requested_day_not_in_future_for_target"
            continue

        forecast_index = pd.date_range(start=forecast_start, end=day_end, freq="h", inclusive="left")
        pred = recursive_forecast_for_index(models[target], history, forecast_index)
        day_pred = pred.loc[(pred.index >= day_start) & (pred.index < day_end)]

        if len(day_pred) != 24:
            skipped_targets[target] = "could_not_form_full_day_prediction"
            continue

        aggregate += day_pred.to_numpy(dtype=float)
        used_targets.append(target)

    if not used_targets:
        raise ValueError("No targets produced a valid prediction for the requested date.")

    hourly = [
        {
            "timestamp": ts.isoformat(),
            "aggregate_usage": float(val),
        }
        for ts, val in zip(day_hours, aggregate)
    ]

    return {
        "date": day_start.date().isoformat(),
        "daily_total": float(np.sum(aggregate)),
        "hourly": hourly,
        "used_targets": used_targets,
        "skipped_targets": skipped_targets,
    }


def _predict_excel_rows_for_day(day_start: pd.Timestamp):
    result = _predict_aggregate_for_day(day_start)
    rows = []
    for item in result["hourly"]:
        ts = pd.Timestamp(item["timestamp"])
        rows.append(
            {
                "hour": ts.strftime("%H:%M"),
                "predicted_consumption": float(item["aggregate_usage"]),
            }
        )

    return {
        "date": result["date"],
        "hourly": rows,
        "daily_total": result["daily_total"],
        "used_targets": result["used_targets"],
        "skipped_targets": result["skipped_targets"],
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict-day", methods=["GET", "POST"])
def predict_day():
    try:
        payload = request.get_json(silent=True) or {}
        print(f"/predict-day : {payload}")
        
        date_value = payload.get("date") or request.args.get("date")
        if date_value is not None:
            date_value = str(date_value)
        day_start = _parse_input_date(date_value)
        result = _predict_aggregate_for_day(day_start)
        return jsonify(result)
    except Exception as exc:
        print(f"error: {exc}")
        return jsonify({"error": str(exc)}), 400


@app.route("/predict-day-excel", methods=["GET", "POST"])
def predict_day_excel():
    try:
        payload = request.get_json(silent=True) or {}
        date_value = payload.get("date") or request.args.get("date")
        if date_value is not None:
            date_value = str(date_value)

        day_start = _parse_input_date(date_value)
        result = _predict_excel_rows_for_day(day_start)

        # Use format=csv for direct Excel/PowerQuery import as two columns.
        output_format = (request.args.get("format") or payload.get("format") or "json").lower()
        if output_format == "csv":
            df_out = pd.DataFrame(result["hourly"], columns=["hour", "predicted_consumption"])
            csv_buffer = StringIO()
            df_out.to_csv(csv_buffer, index=False)
            return Response(
                csv_buffer.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=prediction_{result['date']}.csv"
                },
            )

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
