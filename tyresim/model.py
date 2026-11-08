"""Modelos de Scikit-Learn para predecir el tiempo de vuelta."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

FEATURES = ["compound", "tyre_age", "fuel_kg", "track_temp"]
TARGET = "lap_time"


def build_pipeline(kind: str = "gbr") -> Pipeline:
    """`ridge`: regresión polinómica (interpretable). `gbr`: gradient boosting (no lineal)."""
    if kind == "ridge":
        numeric = Pipeline([("poly", PolynomialFeatures(degree=3, include_bias=False)),
                            ("scale", StandardScaler())])
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["compound"]),
            ("num", numeric, ["tyre_age", "fuel_kg", "track_temp"]),
        ])
        # Interacciones compuesto x edad: cada compuesto se degrada distinto
        return Pipeline([("pre", pre), ("inter", PolynomialFeatures(degree=2, interaction_only=True)),
                         ("model", Ridge(alpha=1.0))])
    if kind == "gbr":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["compound"]),
            ("num", "passthrough", ["tyre_age", "fuel_kg", "track_temp"]),
        ])
        return Pipeline([("pre", pre),
                         ("model", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                                                 random_state=0))])
    raise ValueError(f"Modelo desconocido: {kind}")


@dataclass
class CVResult:
    kind: str
    mae: float
    rmse: float
    r2: float


def cross_validate(df: pd.DataFrame, kind: str = "gbr", n_splits: int = 5) -> CVResult:
    """Validación cruzada agrupada por carrera (evita fuga de información entre vueltas de la misma carrera)."""
    X, y, groups = df[FEATURES], df[TARGET].to_numpy(), df["race_id"].to_numpy()
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pipe = build_pipeline(kind).fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(root_mean_squared_error(y[test_idx], pred))
        r2s.append(r2_score(y[test_idx], pred))
    return CVResult(kind, float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(r2s)))


def train(df: pd.DataFrame, kind: str = "gbr") -> Pipeline:
    return build_pipeline(kind).fit(df[FEATURES], df[TARGET])


def make_lap_time_fn(pipe: Pipeline):
    """Adapta el modelo a la interfaz del simulador: f(compound, ages, fuel, temp) -> np.ndarray."""
    def fn(compound, tyre_age, fuel_kg, track_temp):
        n = len(tyre_age)
        X = pd.DataFrame({"compound": [compound] * n, "tyre_age": tyre_age,
                          "fuel_kg": fuel_kg, "track_temp": np.full(n, track_temp)})
        return pipe.predict(X)
    return fn
