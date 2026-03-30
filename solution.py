import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import time
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os
import holidays
import numpy as np
import itertools
from pathlib import Path
from collections import deque
from queue import PriorityQueue

# CONSTANTS
FEATURES = ['hour', 'dayofweek', 'hour_of_week', 'month', 'dayofyear',
            'lag_1', 'lag_168',
            'diff_1', 'diff_6', 'diff_12',  'diff_24', 'diff_48',
            'rolling_std_24h', 'rolling_max_24h', 'rolling_min_24h',
            'ema_24h', 'ema_std_12h', 'ema_7d_same_hour',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
            'is_holiday', 'is_day_before_holiday', 'is_day_after_holiday']

class Data:
    def __init__(self, X_cv, y_cv, X_test, y_test, X_all, y_all):
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.X_test = X_test
        self.y_test = y_test
        self.X_all = X_all
        self.y_all = y_all

def create_features(data: pd.DataFrame, target: str):
    df_feat = pd.DataFrame()
    df_feat.index = data.index.copy()
    df_feat['value'] = data[target].copy()
    
    # Base Temporal features
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['hour_of_week'] = df_feat['dayofweek'] * 24 + df_feat['hour']
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear
    
    # Cyclical Encoding (Sine/Cosine)
    # 24 hours in a day
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
    
    # 12 months in a year
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    # 366 days to safely cover leap years
    df_feat['dayofyear_sin'] = np.sin(2 * np.pi * df_feat['dayofyear'] / 366)
    df_feat['dayofyear_cos'] = np.cos(2 * np.pi * df_feat['dayofyear'] / 366)
    
    # Holiday Flags
    years = df_feat.index.year.unique().tolist()
    
    # Convert holiday keys to pandas timestamps and normalize to day-level
    us_holidays = pd.to_datetime(list(holidays.US(years=years).keys())).normalize()

    # Build a day-level, tz-naive view of the index to match holiday timestamps reliably
    dates = pd.DatetimeIndex(df_feat.index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()

    df_feat['is_holiday'] = dates.isin(us_holidays).astype(int)
    
    # Apply the same fix to the before/after flags
    df_feat['is_day_before_holiday'] = (dates + pd.Timedelta(days=1)).isin(us_holidays).astype(int)
    df_feat['is_day_after_holiday'] = (dates - pd.Timedelta(days=1)).isin(us_holidays).astype(int)
    
    # Autoregressive Lags
    df_feat['lag_1'] = df_feat['value'].shift(1)
    df_feat['lag_168'] = df_feat['value'].shift(168)
    df_feat['diff_1'] = df_feat['lag_1'] - df_feat['value'].shift(2)
    #df_feat['acc_1'] = df_feat['value'].shift(2) - df_feat['value'].shift(3)
    df_feat['diff_6'] = df_feat['value'].shift(6) - df_feat['value'].shift(7)
    df_feat['diff_12'] = df_feat['value'].shift(12) - df_feat['value'].shift(13)
    df_feat['diff_24'] = df_feat['value'].shift(24) - df_feat['value'].shift(25)
    df_feat['diff_48'] = df_feat['value'].shift(48) - df_feat['value'].shift(49)
    
    # Rolling Statistics
    df_feat['rolling_std_24h'] = df_feat['value'].shift(1).rolling(window=24).std()
    df_feat['rolling_max_24h'] = df_feat['value'].shift(1).rolling(window=24).max()
    df_feat['rolling_min_24h'] = df_feat['value'].shift(1).rolling(window=24).min()

    # Exponential moving averages
    df_feat['ema_24h'] = df_feat['value'].shift(1).ewm(span=24, adjust=False).mean()
    df_feat['ema_std_12h'] = df_feat['value'].shift(1).ewm(span=12, adjust=False).std(bias=False)
    df_feat['ema_7d_same_hour'] = (
        df_feat.groupby(df_feat.index.hour)['value']
        .transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean())
    )
    
    return df_feat.dropna()

def get_data():
    global FEATURES
    # Load Data
    df = pd.read_parquet('data/est_hourly.parquet')

    # Ensure index is datetime and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    
    for target in df.keys():
        start_time = time.time()
        df_features = create_features(df, target)
        end_time = time.time()
        
        print(f"{target}: Feature extraction time: {end_time - start_time}s")

        # Time-based CV pool and strict holdout test split (last 1 year).
        last_ts = df_features.index.max()
        test_start = last_ts - pd.DateOffset(years=1) + pd.Timedelta(hours=1)
        cv_pool = df_features.loc[df_features.index < test_start]
        test = df_features.loc[df_features.index >= test_start]
        all_data = df_features

        if cv_pool.empty or test.empty:
            raise ValueError(
                f"Invalid split for {target}. CV rows={len(cv_pool)}, Test rows={len(test)}"
            )

        print(
            f"CV pool rows: {len(cv_pool)}, Test rows: {len(test)} | "
            f"Test window: {test.index.min()} -> {test.index.max()}"
        )
        
        yield Data(
            cv_pool[FEATURES],
            cv_pool['value'],
            test[FEATURES],
            test['value'],
            all_data[FEATURES],
            all_data['value']
        ), test.index, target

def train_test_model(X_train, y_train, X_val, y_val, X_test, depth, lr, subsample, n_estimators=1000):
    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0
    )
    
    pred_test = reg.predict(X_test)

    return reg, pred_test

def time_series_cv_rmse(data: Data, depth, lr, subsample, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    for train_idx, val_idx in tscv.split(data.X_cv):
        X_train_fold = data.X_cv.iloc[train_idx]
        y_train_fold = data.y_cv.iloc[train_idx]
        X_val_fold = data.X_cv.iloc[val_idx]
        y_val_fold = data.y_cv.iloc[val_idx]

        model, _ = train_test_model(
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            X_val_fold,
            depth,
            lr,
            subsample
        )

        # Score folds with recursive roll-forward to match final evaluation.
        val_pred_recursive = recursive_forecast_for_index(
            model,
            y_train_fold,
            X_val_fold.index,
        )
        fold_rmses.append(root_mean_squared_error(y_val_fold, val_pred_recursive))

    return float(np.mean(fold_rmses)), float(np.std(fold_rmses))

def train_best_on_cv_pool_and_test(data: Data, depth, lr, subsample):
    # Use a chronological tail only to determine best boosting rounds.
    split_idx = int(len(data.X_cv) * 0.9)
    X_train_probe = data.X_cv.iloc[:split_idx]
    y_train_probe = data.y_cv.iloc[:split_idx]
    X_val_probe = data.X_cv.iloc[split_idx:]
    y_val_probe = data.y_cv.iloc[split_idx:]

    probe_model, _ = train_test_model(
        X_train_probe,
        y_train_probe,
        X_val_probe,
        y_val_probe,
        X_val_probe,
        depth,
        lr,
        subsample
    )

    best_rounds = int(getattr(probe_model, 'best_iteration', 999)) + 1
    best_rounds = max(50, best_rounds)

    # Retrain on full pre-2017 history using the selected number of trees.
    model = xgb.XGBRegressor(
        n_estimators=best_rounds,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    model.fit(data.X_cv, data.y_cv, verbose=0)

    pred_test = model.predict(data.X_test)
    test_rmse = root_mean_squared_error(data.y_test, pred_test)

    print(f"Selected boosting rounds: {best_rounds}")

    return model, pred_test, test_rmse

def train_final_on_all_history(data: Data, depth, lr, subsample):
    # Keep a chronological tail from full history for early stopping.
    split_idx = int(len(data.X_all) * 0.9)
    X_train = data.X_all.iloc[:split_idx]
    y_train = data.y_all.iloc[:split_idx]
    X_val = data.X_all.iloc[split_idx:]
    y_val = data.y_all.iloc[split_idx:]

    model, _ = train_test_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_val,
        depth,
        lr,
        subsample
    )
    return model

def recursive_forecast_for_index(model, history_target: pd.Series, forecast_index: pd.DatetimeIndex):
    history = history_target.copy().sort_index()
    forecast_index = pd.DatetimeIndex(forecast_index).sort_values()

    if len(history) < 168:
        raise ValueError("Need at least 168 historical points for lag_168 feature.")

    years = sorted(set(forecast_index.year.tolist()))
    us_holidays = set(pd.to_datetime(list(holidays.US(years=years).keys())).normalize())

    # Precompute rolling structures so each forecast step is O(1).
    history_values = history.tolist()
    last_24 = deque(history_values[-24:], maxlen=24)
    last_168 = deque(history_values[-168:], maxlen=168)
    sum_last_24 = float(sum(last_24))

    hour_windows = {h: deque(maxlen=7) for h in range(24)}
    for ts_hist, val_hist in history.items():
        hour_windows[ts_hist.hour].append(float(val_hist))

    # Initialize EMA states from observed history.
    alpha_24h = 2.0 / (24 + 1)
    ema_24h_state = float(history.ewm(span=24, adjust=False).mean().iloc[-1])

    alpha_12h = 2.0 / (12 + 1)
    ema_mean_12_state = float(history.ewm(span=12, adjust=False).mean().iloc[-1])
    ema_std_12_state = float(history.ewm(span=12, adjust=False).std(bias=False).iloc[-1])
    if np.isnan(ema_std_12_state):
        ema_std_12_state = float(np.std(history_values[-12:]))
    
    alpha_7h = 2.0 / (7 + 1)
    hour_ema_state = {}
    for h in range(24):
        hour_hist = history[history.index.hour == h]
        hour_ema_state[h] = float(hour_hist.ewm(span=7, adjust=False).mean().iloc[-1])

    preds = []
    total_steps = len(forecast_index)
    loop_start = time.time()
    for i, ts in enumerate(forecast_index, start=1):
        day = ts.normalize()
        hour = ts.hour
        month = ts.month
        dayofyear = ts.dayofyear

        ema_7d_same_hour = hour_ema_state[hour]

        feat_row = {
            'hour': hour,
            'dayofweek': ts.dayofweek,
            'hour_of_week': ts.dayofweek * 24 + hour,
            'month': month,
            'dayofyear': dayofyear,
            'lag_1': last_168[-1],
            'lag_168': last_168[-168],
            'diff_1': last_168[-1] - last_168[-2],
            #'acc_1': (last_168[-1] - last_168[-2]) - (last_168[-2] - last_168[-3]),
            'diff_6': last_168[-6] - last_168[-7],
            'diff_12': last_168[-12] - last_168[-13],
            'diff_24': last_168[-24] - last_168[-25],
            'diff_48': last_168[-48] - last_168[-49],
            'rolling_std_24h': float(np.std(last_24, ddof=1)),
            'rolling_max_24h': float(np.max(last_24)),
            'rolling_min_24h': float(np.min(last_24)),
            'ema_24h': ema_24h_state,
            'ema_std_12h': ema_std_12_state,
            'ema_7d_same_hour': ema_7d_same_hour,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'dayofyear_sin': np.sin(2 * np.pi * dayofyear / 366),
            'dayofyear_cos': np.cos(2 * np.pi * dayofyear / 366),
            'is_holiday': int(day in us_holidays),
            'is_day_before_holiday': int((day + pd.Timedelta(days=1)) in us_holidays),
            'is_day_after_holiday': int((day - pd.Timedelta(days=1)) in us_holidays),
        }

        x_next = pd.DataFrame([feat_row], index=[ts])[FEATURES]
        pred = float(model.predict(x_next)[0])
        preds.append(pred)

        # Update rolling buffers.
        if len(last_24) == last_24.maxlen:
            sum_last_24 -= last_24[0]
        last_24.append(pred)
        sum_last_24 += pred
        last_168.append(pred)
        hour_windows[hour].append(pred)

        # Online EWMA variance/std update (adjust=False compatible recursion).
        delta = pred - ema_mean_12_state
        ema_mean_12_state = (1.0 - alpha_12h) * ema_mean_12_state + alpha_12h * pred
        ew_var = (1.0 - alpha_12h) * (ema_std_12_state ** 2 + alpha_12h * (delta ** 2))
        ema_std_12_state = float(np.sqrt(max(ew_var, 0.0)))

        ema_24h_state = alpha_24h * pred + (1.0 - alpha_24h) * ema_24h_state
        hour_ema_state[hour] = alpha_7h * pred + (1.0 - alpha_7h) * hour_ema_state[hour]

        if i % 1000 == 0 or i == total_steps:
            elapsed = time.time() - loop_start
            print(f"Recursive forecast progress: {i}/{total_steps} ({elapsed:.1f}s)")

    return pd.Series(preds, index=forecast_index, name='prediction')

def recursive_next_year_forecast(model, history_target: pd.Series):
    future_start = history_target.index.max() + pd.Timedelta(hours=1)
    future_end = history_target.index.max() + pd.DateOffset(years=1)
    future_index = pd.date_range(start=future_start, end=future_end, freq='h', inclusive='left')
    return recursive_forecast_for_index(model, history_target, future_index)

def ensemble_pred(models_dir: str, data: Data, test_index):
    models_pred = []
    models_importance = []
    skipped_models = 0

    for model_file_path in Path(models_dir).iterdir():
        if model_file_path.is_file():
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)

                try:
                    recursive_test_pred = recursive_forecast_for_index(model, data.y_cv, test_index)
                    recursive_test_rmse = root_mean_squared_error(data.y_test, recursive_test_pred)

                    print(f"{model_file_path.name} -> Recursive 2017-2018 RMSE: {recursive_test_rmse:.3f}")
                    models_importance.append(model.feature_importances_)
                    models_pred.append(recursive_test_pred)
                except ValueError as exc:
                    # Skip stale models trained on a different feature schema.
                    if 'feature_names mismatch' in str(exc):
                        skipped_models += 1
                        print(f"Skipping incompatible model {model_file_path.name}: feature schema mismatch")
                        continue
                    raise

    if not models_pred:
        raise ValueError(f"No model files found in {models_dir}")

    if skipped_models > 0:
        print(f"Skipped {skipped_models} incompatible model(s) due to feature mismatch.")

    mean_importance = np.mean(np.array(models_importance), axis=0)
    importance_df = pd.DataFrame({
        'feature': FEATURES,
        'mean_importance': mean_importance
    }).sort_values('mean_importance', ascending=False)

    print("=== Aggregated Feature Importance (Mean Across Ensemble Models) ===")
    print(importance_df.to_string(index=False))
    
    models_pred = np.array(models_pred).mean(axis=0)
    
    return models_pred

def plot(actual, pred, test_index, plot_file):
    print(actual.shape, pred.shape)
    rmse = root_mean_squared_error(actual, pred)
    print(f"RMSE: {rmse}")
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Overlay actual and prediction on the same axis to make differences visible.
    axs[0].plot(test_index, actual, label='Actual', alpha=0.9, color='blue', linewidth=1)
    axs[0].plot(test_index, pred, label='Predicted', alpha=0.8, color='orange', linewidth=1)
    axs[0].set_title(f'Actual vs Predicted (RMSE={rmse:.2f})')
    axs[0].set_ylabel('Load')
    axs[0].legend(loc='upper right')

    # Residual plot: actual - predicted.
    axs[1].plot(test_index, actual - pred, color='crimson', linewidth=0.8)
    axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[1].set_title('Residuals (Actual - Predicted)')
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Datetime')

    plt.tight_layout()

    # Save
    fig.savefig(plot_file, dpi=300)

def get_param_grid_comb():
    # Hyperparameter grid to explore
    param_grid = {
        'max_depth': [5, 7, 10, 12],
        'learning_rate': [0.1, 0.05],
        'subsample': [0.8, 1.0],
    }

    # Generate all combinations
    return list(itertools.product(
        param_grid['max_depth'],
        param_grid['learning_rate'],
        param_grid['subsample']
    ))

def grid_search(data: Data):
    print("=== Hyperparameter Tuning (Rolling Time-Series CV, Recursive Metric) ===\n")
    print(f"{'max_depth':<10} {'lr':<8} {'subsample':<12} {'CV RMSE':<12} {'CV STD':<10} {'Duration(s)':<12}")
    print("-" * 78)

    pq = PriorityQueue()
    for i, (depth, lr, subsample) in enumerate(get_param_grid_comb()):
        start_time = time.time()
        cv_rmse, cv_std = time_series_cv_rmse(data, depth, lr, subsample)
        end_time = time.time()

        elem = {
            'depth': depth,
            'lr': lr,
            'subsample': subsample,
        }

        pq.put((-cv_rmse, elem))
        if pq.qsize() > 3:
            pq.get()

        print(f"{depth:<10} {lr:<8} {subsample:<12} {cv_rmse:<12.2f} {cv_std:<10.2f} {end_time-start_time:<12.2f}")

    return pq

def run_training_pipeline():
    model_name = 'xgboost'
    model_version = 15
    model_root = f"{model_name}/v{model_version}"
    aggregate_plot_file = f"{model_root}/plots/prediction_last_year.png"
    os.makedirs(os.path.dirname(aggregate_plot_file), exist_ok=True)

    # Targets to exclude from aggregate totals (short history / stale coverage)
    # AEP AND COMED WOULD NORMALLY NOT BE IGNORED, THEY ARE NOW CAUSE THEY ARE ALREADY COMPUTED
    AGGREGATE_EXCLUDED_TARGETS = {'pjm_load', 'ni'}

    # Found with grid search on AEP, avoiding grid search for each target as it is quite time consuming
    best_params = {'depth': 7, 'lr': 0.05, 'subsample': 0.8}
    aggregate_pred = None
    aggregate_actual = None
    aggregate_test_index = None

    for data, test_index, target in get_data():
        include_in_aggregate = target.lower() not in AGGREGATE_EXCLUDED_TARGETS
        if not include_in_aggregate:
            print(f"Skipping {target} in aggregate computation by config.")
            
        # Create directories
        model_target_root = f"{model_root}/{target}"
        models_dir = f"{model_target_root}/models/"
        model_file = f"{models_dir}/model"
        plot_file = f"{model_target_root}/plots/prediction_last_year.png"
        forecast_file = f"{model_target_root}/predictions/_next_year_hourly.csv"

        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(forecast_file), exist_ok=True)

        # best_params_pq = grid_search(data)
        # while not best_params_pq.empty():
        #     cv_rmse, best_params = best_params_pq.get()
        #     cv_rmse = -cv_rmse

        # Train model until 2017 and use the rest for testing
        best_model, pred, test_rmse = train_best_on_cv_pool_and_test(
            data,
            best_params['depth'],
            best_params['lr'],
            best_params['subsample']
        )
        # Save partial model
        # print(f"best_params: {best_params}, test_rmse: {test_rmse}, cv_rmse: {cv_rmse}")
        pickle.dump(best_model, open(f"{model_file}_partial_{best_params['depth']}_{best_params['lr']}_{best_params['subsample']}", 'wb'))
        # print(f"\nTeacher-forced last-year RMSE: {test_rmse:.3f}")

        # Recursive backtest for the last-year holdout.
        test_pred = recursive_forecast_for_index(best_model, data.y_cv, test_index)
        test_rmse = root_mean_squared_error(data.y_test, test_pred)
        print(f"Recursive last-year RMSE: {test_rmse:.3f}")
        plot(data.y_test, test_pred, test_index, plot_file)
        print(f"Saved recursive backtest plot -> {plot_file}")

        # Train model on all the data
        final_model = train_final_on_all_history(
            data,
            best_params['depth'],
            best_params['lr'],
            best_params['subsample']
        )
        pickle.dump(final_model, open(f"{model_file}", 'wb'))

        if include_in_aggregate:
            # Initialize aggregate container using the first included target.
            if aggregate_test_index is None:
                aggregate_test_index = test_index
                aggregate_pred = np.zeros(len(test_index), dtype=float)
                aggregate_actual = np.zeros(len(test_index), dtype=float)

            if len(test_index) != len(aggregate_test_index) or not test_index.equals(aggregate_test_index):
                raise ValueError(
                    f"Test index mismatch for {target}. Aggregation requires aligned hourly windows."
                )

            aggregate_pred += test_pred.to_numpy(dtype=float)
            aggregate_actual += data.y_test.to_numpy(dtype=float)

    if aggregate_test_index is None:
        raise ValueError("No targets available for aggregate computation after exclusions.")

    plot(aggregate_actual, aggregate_pred, aggregate_test_index, aggregate_plot_file)
    print(f"Saved recursive backtest plot -> {aggregate_plot_file}")


if __name__ == '__main__':
    run_training_pipeline()