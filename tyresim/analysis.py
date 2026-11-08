"""Limpieza de datos y análisis de degradación con Pandas y NumPy."""
import numpy as np
import pandas as pd

from .config import DEFAULT_TRACK, Track


def clean_laps(df: pd.DataFrame, threshold: float = 1.07) -> pd.DataFrame:
    """Quita vueltas de boxes, tráfico y outliers (>107 % de la mediana de la carrera)."""
    out = df[~df["pit_in"] & ~df["pit_out"] & ~df["traffic"]].copy()
    median = out.groupby("race_id")["lap_time"].transform("median")
    return out[out["lap_time"] < median * threshold].reset_index(drop=True)


def fuel_correct(df: pd.DataFrame, track: Track = DEFAULT_TRACK) -> pd.DataFrame:
    """Añade `lap_time_fc`: tiempo corregido por el peso del combustible."""
    out = df.copy()
    out["lap_time_fc"] = out["lap_time"] - out["fuel_kg"] * track.fuel_effect
    return out


def add_stint_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Delta respecto a las primeras vueltas del stint -> aísla el desgaste del ritmo del piloto."""
    out = df.copy()
    keys = ["race_id", "driver", "stint"]
    ref = (out[out["tyre_age"] <= 3].groupby(keys)["lap_time_fc"].mean().rename("stint_ref"))
    out = out.join(ref, on=keys)
    out["deg_delta"] = out["lap_time_fc"] - out["stint_ref"]
    return out.dropna(subset=["deg_delta"])


def degradation_by_compound(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """Ajusta un polinomio (np.polyfit) de deg_delta vs tyre_age para cada compuesto."""
    results = []
    for compound, g in df.groupby("compound", observed=True):
        x, y = g["tyre_age"].to_numpy(), g["deg_delta"].to_numpy()
        coefs = np.polyfit(x, y, degree)
        pred = np.polyval(coefs, x)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        lin_slope = np.polyfit(x[x <= 15], y[x <= 15], 1)[0]
        results.append({"compound": compound, "deg_s_per_lap": lin_slope, "r2": r2,
                        "n_laps": len(g), "coefs": coefs})
    return pd.DataFrame(results).set_index("compound").sort_values("deg_s_per_lap")


def stint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen por stint: longitud, ritmo medio y degradación (pendiente lineal)."""
    def slope(g):
        return np.polyfit(g["tyre_age"], g["lap_time_fc"], 1)[0] if len(g) > 3 else np.nan

    keys = ["race_id", "driver", "stint", "compound"]
    agg = df.groupby(keys, observed=True).agg(
        laps=("lap", "count"), mean_lap=("lap_time_fc", "mean"),
        best_lap=("lap_time_fc", "min"), track_temp=("track_temp", "first"))
    agg["deg_slope"] = df.groupby(keys, observed=True)[["tyre_age", "lap_time_fc"]].apply(slope)
    return agg.reset_index()


def detect_cliff(df: pd.DataFrame, fit_laps: int = 10, threshold: float = 0.8,
                 window: int = 3) -> pd.DataFrame:
    """Estima la vuelta del 'cliff' de cada stint.

    1. Ajusta una recta a las primeras `fit_laps` vueltas (degradación "normal").
    2. Calcula el residuo del resto del stint respecto a esa recta.
    3. El cliff es la primera vuelta donde la media móvil del residuo supera `threshold` s.
    Los stints que no llegan al cliff no aparecen en el resultado.
    """
    keys = ["race_id", "driver", "stint", "compound"]

    def first_cliff(g: pd.DataFrame) -> float:
        g = g.sort_values("tyre_age")
        early = g[g["tyre_age"] <= fit_laps]
        if len(early) < 5 or len(g) <= fit_laps + window:
            return np.nan
        slope, intercept = np.polyfit(early["tyre_age"], early["lap_time_fc"], 1)
        resid = g["lap_time_fc"] - (slope * g["tyre_age"] + intercept)
        smooth = resid.rolling(window).mean()
        hits = g.loc[smooth > threshold, "tyre_age"]
        # restamos window-1 porque la media móvil "llega tarde"
        return float(hits.iloc[0] - (window - 1)) if len(hits) else np.nan

    cliffs = df.groupby(keys, observed=True)[["tyre_age", "lap_time_fc"]].apply(first_cliff)
    return cliffs.rename("cliff_lap").dropna().reset_index()


def temperature_effect(stints: pd.DataFrame, bins=(20, 30, 40, 55)) -> pd.DataFrame:
    """Tabla pivote: degradación media por compuesto y rango de temperatura de pista."""
    s = stints.copy()
    s["temp_range"] = pd.cut(s["track_temp"], bins=bins)
    return s.pivot_table(index="compound", columns="temp_range", values="deg_slope",
                         aggfunc="mean", observed=True).round(3)
