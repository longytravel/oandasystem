"""
ML Exit Model Training.

Trains two gradient-boosting models for the exit strategy:
1. hold_value_model: regression on future_r_change_5bar (predicts value of holding)
2. adverse_risk_model: binary classification on hit_sl_before_tp (predicts SL risk)

Library priority: CatBoost > LightGBM > sklearn GradientBoosting.
Hyperparameters tuned via Optuna with time-series cross-validation.
"""
import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import optuna

from pipeline.ml_exit.features import get_feature_names

# Silence Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Detect available ML backend ---
_ML_BACKEND = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
    _ML_BACKEND = 'catboost'
except ImportError:
    pass

if _ML_BACKEND is None:
    try:
        import lightgbm as lgb
        _ML_BACKEND = 'lightgbm'
    except ImportError:
        pass

if _ML_BACKEND is None:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        _ML_BACKEND = 'sklearn'
    except ImportError:
        pass


def get_ml_backend() -> Optional[str]:
    """Return the active ML backend name, or None if nothing available."""
    return _ML_BACKEND


@dataclass
class MLExitModels:
    """Container for trained ML exit models and metadata."""
    hold_value_model: Any = None          # Regression model
    adverse_risk_model: Any = None        # Classification model
    feature_names: List[str] = field(default_factory=get_feature_names)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    backend: str = ''


# ---------------------------------------------------------------------------
# Time-series cross-validation splits
# ---------------------------------------------------------------------------

def _ts_cv_splits(
    n_samples: int,
    n_folds: int = 5,
    min_train_frac: float = 0.3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time-series aware CV splits (expanding window).

    Each fold uses all data up to a cutoff for training and
    the next block for validation. No shuffling — temporal order preserved.

    Args:
        n_samples: Total number of samples
        n_folds: Number of validation folds
        min_train_frac: Minimum fraction of data for first training set

    Returns:
        List of (train_indices, val_indices) tuples
    """
    min_train = max(int(n_samples * min_train_frac), 30)
    remaining = n_samples - min_train
    fold_size = max(remaining // (n_folds + 1), 10)

    splits = []
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        val_start = train_end
        val_end = min(val_start + fold_size, n_samples)
        if val_start >= n_samples:
            break
        splits.append((
            np.arange(0, train_end),
            np.arange(val_start, val_end),
        ))
    return splits


# ---------------------------------------------------------------------------
# Sklearn backend
# ---------------------------------------------------------------------------

def _train_sklearn_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
) -> Tuple[Any, Dict[str, float]]:
    """Train sklearn GradientBoostingRegressor with Optuna tuning."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            }
            model = GradientBoostingRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = GradientBoostingRegressor(
        **study.best_params, random_state=42
    )
    best_model.fit(X_train, y_train)

    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'val_r2': float(r2_score(y_val, y_pred_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


def _train_sklearn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
) -> Tuple[Any, Dict[str, float]]:
    """Train sklearn GradientBoostingClassifier with Optuna tuning."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss

    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            }
            model = GradientBoostingClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            return -auc if np.isfinite(auc) else 0.0  # Minimize negative AUC
        except Exception:
            return 0.0  # Worst AUC (0.5 random) as penalty

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = GradientBoostingClassifier(
        **study.best_params, random_state=42
    )
    best_model.fit(X_train, y_train)

    y_prob_train = best_model.predict_proba(X_train)[:, 1]
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        'train_auc': float(roc_auc_score(y_train, y_prob_train)),
        'val_auc': float(roc_auc_score(y_val, y_prob_val)),
        'train_logloss': float(log_loss(y_train, y_prob_train)),
        'val_logloss': float(log_loss(y_val, y_prob_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


# ---------------------------------------------------------------------------
# CatBoost backend
# ---------------------------------------------------------------------------

def _train_catboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
    early_stopping_rounds: int = 20,
) -> Tuple[Any, Dict[str, float]]:
    """Train CatBoostRegressor with Optuna tuning."""
    from catboost import CatBoostRegressor, Pool
    from sklearn.metrics import mean_squared_error, r2_score

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    def objective(trial):
        try:
            params = {
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 50, 500),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_seed': 42,
                'verbose': 0,
            }
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = CatBoostRegressor(
        **{k: v for k, v in study.best_params.items()},
        random_seed=42, verbose=0,
    )
    best_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'val_r2': float(r2_score(y_val, y_pred_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


def _train_catboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
    early_stopping_rounds: int = 20,
) -> Tuple[Any, Dict[str, float]]:
    """Train CatBoostClassifier with Optuna tuning."""
    from catboost import CatBoostClassifier, Pool
    from sklearn.metrics import roc_auc_score, log_loss

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    def objective(trial):
        try:
            params = {
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 50, 500),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_seed': 42,
                'verbose': 0,
            }
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            return -auc if np.isfinite(auc) else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = CatBoostClassifier(
        **{k: v for k, v in study.best_params.items()},
        random_seed=42, verbose=0,
    )
    best_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

    y_prob_train = best_model.predict_proba(X_train)[:, 1]
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        'train_auc': float(roc_auc_score(y_train, y_prob_train)),
        'val_auc': float(roc_auc_score(y_val, y_prob_val)),
        'train_logloss': float(log_loss(y_train, y_prob_train)),
        'val_logloss': float(log_loss(y_val, y_prob_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


# ---------------------------------------------------------------------------
# LightGBM backend
# ---------------------------------------------------------------------------

def _train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
    early_stopping_rounds: int = 20,
) -> Tuple[Any, Dict[str, float]]:
    """Train LightGBM regressor with Optuna tuning."""
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, r2_score

    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'random_state': 42,
                'verbosity': -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse if np.isfinite(mse) else 1e6
        except Exception:
            return 1e6

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = lgb.LGBMRegressor(
        **study.best_params, random_state=42, verbosity=-1,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'val_rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'val_r2': float(r2_score(y_val, y_pred_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


def _train_lightgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_optuna_trials: int = 30,
    early_stopping_rounds: int = 20,
) -> Tuple[Any, Dict[str, float]]:
    """Train LightGBM classifier with Optuna tuning."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, log_loss

    def objective(trial):
        try:
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'random_state': 42,
                'verbosity': -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            return -auc if np.isfinite(auc) else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)

    best_model = lgb.LGBMClassifier(
        **study.best_params, random_state=42, verbosity=-1,
    )
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    y_prob_train = best_model.predict_proba(X_train)[:, 1]
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        'train_auc': float(roc_auc_score(y_train, y_prob_train)),
        'val_auc': float(roc_auc_score(y_val, y_prob_val)),
        'train_logloss': float(log_loss(y_train, y_prob_train)),
        'val_logloss': float(log_loss(y_val, y_prob_val)),
        'best_params': study.best_params,
    }
    return best_model, metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_ml_exit_models(
    train_df: pd.DataFrame,
    n_optuna_trials: int = 30,
    cv_folds: int = 5,
    early_stopping_rounds: int = 20,
) -> Optional[MLExitModels]:
    """
    Train ML exit models (hold-value regressor + adverse-risk classifier).

    Automatically selects best available backend (CatBoost > LightGBM > sklearn).
    Uses Optuna for hyperparameter tuning with time-series cross-validation.

    Args:
        train_df: DataFrame with feature columns (from get_feature_names())
                  plus label columns 'future_r_change_5bar' and 'hit_sl_before_tp'.
        n_optuna_trials: Number of Optuna trials for hyperparameter search.
        cv_folds: Number of time-series CV folds.
        early_stopping_rounds: Early stopping patience (CatBoost/LightGBM only).

    Returns:
        MLExitModels with trained models, feature importances, and metrics.

    Raises:
        RuntimeError: If no ML backend is available.
    """
    from loguru import logger as ml_logger

    backend = get_ml_backend()
    if backend is None:
        raise RuntimeError(
            'No ML backend available. Install catboost, lightgbm, or scikit-learn.'
        )

    feature_names = get_feature_names()

    # Validate required columns
    missing_features = [f for f in feature_names if f not in train_df.columns]
    if missing_features:
        raise ValueError(f'Missing feature columns: {missing_features}')

    # V2: Use remaining_pnl_r as primary regression target (optimal stopping)
    # Falls back to future_r_change_5bar for V1 datasets
    reg_target = 'remaining_pnl_r' if 'remaining_pnl_r' in train_df.columns else 'future_r_change_5bar'
    if reg_target not in train_df.columns:
        raise ValueError(f"Missing label column '{reg_target}'")
    if 'hit_sl_before_tp' not in train_df.columns:
        raise ValueError("Missing label column 'hit_sl_before_tp'")

    X = train_df[feature_names].values.astype(np.float64)
    y_reg = train_df[reg_target].values.astype(np.float64)
    y_cls = train_df['hit_sl_before_tp'].values.astype(np.int32)

    # Handle NaN/inf
    nan_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_reg) & np.isfinite(y_cls.astype(np.float64))
    if nan_mask.sum() < len(X):
        n_dropped = len(X) - nan_mask.sum()
        warnings.warn(f'Dropping {n_dropped} rows with NaN/inf values')
        X = X[nan_mask]
        y_reg = y_reg[nan_mask]
        y_cls = y_cls[nan_mask]

    n_samples = len(X)

    # Fix 6: Graceful skip for low-sample windows (CatBoost CV produces NaN with <50 rows)
    if n_samples < 50:
        ml_logger.warning(f'ML training skipped: only {n_samples} samples (need >= 50)')
        return None

    if n_samples < 100:
        ml_logger.warning(f'Low sample count ({n_samples}), reducing CV folds to 3')
        cv_folds = min(cv_folds, 3)

    # Time-series CV: use last fold as the main validation set
    splits = _ts_cv_splits(n_samples, n_folds=cv_folds)
    if not splits:
        ml_logger.warning(f'ML training skipped: could not create CV splits for {n_samples} samples')
        return None

    # Use last split for train/val
    train_idx, val_idx = splits[-1]
    X_train, X_val = X[train_idx], X[val_idx]
    y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]
    y_cls_train, y_cls_val = y_cls[train_idx], y_cls[val_idx]

    # Ensure both classes present in train and val for classifier
    if len(np.unique(y_cls_train)) < 2 or len(np.unique(y_cls_val)) < 2:
        # Fall back to a 70/30 split preserving temporal order
        split_point = int(n_samples * 0.7)
        X_train, X_val = X[:split_point], X[split_point:]
        y_reg_train, y_reg_val = y_reg[:split_point], y_reg[split_point:]
        y_cls_train, y_cls_val = y_cls[:split_point], y_cls[split_point:]

    # --- Train hold-value regressor ---
    if backend == 'catboost':
        hold_model, hold_metrics = _train_catboost_regressor(
            X_train, y_reg_train, X_val, y_reg_val,
            n_optuna_trials, early_stopping_rounds,
        )
    elif backend == 'lightgbm':
        hold_model, hold_metrics = _train_lightgbm_regressor(
            X_train, y_reg_train, X_val, y_reg_val,
            n_optuna_trials, early_stopping_rounds,
        )
    else:
        hold_model, hold_metrics = _train_sklearn_regressor(
            X_train, y_reg_train, X_val, y_reg_val,
            n_optuna_trials,
        )

    # --- Train adverse-risk classifier ---
    if backend == 'catboost':
        risk_model, risk_metrics = _train_catboost_classifier(
            X_train, y_cls_train, X_val, y_cls_val,
            n_optuna_trials, early_stopping_rounds,
        )
    elif backend == 'lightgbm':
        risk_model, risk_metrics = _train_lightgbm_classifier(
            X_train, y_cls_train, X_val, y_cls_val,
            n_optuna_trials, early_stopping_rounds,
        )
    else:
        risk_model, risk_metrics = _train_sklearn_classifier(
            X_train, y_cls_train, X_val, y_cls_val,
            n_optuna_trials,
        )

    # --- Extract feature importances ---
    if hasattr(hold_model, 'feature_importances_'):
        importances = hold_model.feature_importances_
    elif hasattr(hold_model, 'get_feature_importance'):
        importances = hold_model.get_feature_importance()
    else:
        importances = np.zeros(len(feature_names))

    # Normalize to sum to 1
    total = importances.sum()
    if total > 0:
        importances = importances / total

    feature_importances = {
        name: float(imp)
        for name, imp in zip(feature_names, importances)
    }

    return MLExitModels(
        hold_value_model=hold_model,
        adverse_risk_model=risk_model,
        feature_names=feature_names,
        feature_importances=feature_importances,
        training_metrics={
            'backend': backend,
            'n_samples': n_samples,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'hold_value': hold_metrics,
            'adverse_risk': risk_metrics,
        },
        backend=backend,
    )
