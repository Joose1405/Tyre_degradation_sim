"""Gráficas con Matplotlib (se guardan en outputs/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"SOFT": "#d62728", "MEDIUM": "#e6b800", "HARD": "#7f7f7f"}


def plot_degradation(df: pd.DataFrame, deg: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for compound, g in df.groupby("compound", observed=True):
        m = g.groupby("tyre_age")["deg_delta"].median()
        ax.scatter(m.index, m.values, s=12, color=COLORS[compound], alpha=0.6)
        x = np.linspace(1, g["tyre_age"].max(), 100)
        ax.plot(x, np.polyval(deg.loc[compound, "coefs"], x), color=COLORS[compound],
                label=f"{compound} ({deg.loc[compound, 'deg_s_per_lap']:.3f} s/vuelta)")
    ax.set(xlabel="Vueltas de uso del neumático", ylabel="Pérdida vs. inicio de stint (s)",
           title="Degradación por compuesto (corregida por combustible)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_model_vs_truth(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for compound, g in curves.groupby("compound"):
        ax.plot(g["tyre_age"], g["truth"], color=COLORS[compound], ls="--", alpha=0.7)
        ax.plot(g["tyre_age"], g["pred"], color=COLORS[compound], label=compound)
    ax.set(xlabel="Vueltas de uso", ylabel="Tiempo de vuelta (s)",
           title="Modelo ML (línea) vs. física real (discontinua), combustible fijo")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_strategies(ranking: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Diferencia PAREADA: misma simulación (mismos safety cars) -> se ve el efecto de la estrategia
    best = ranking["_samples"].iloc[0]
    data = [s - best for s in ranking["_samples"]]
    ax.boxplot(data, vert=False, tick_labels=ranking["label"], showfliers=False)
    ax.invert_yaxis()
    ax.set(xlabel="Segundos perdidos vs. la mejor estrategia (misma simulación)",
           title="Monte Carlo de estrategias (ruido + safety car)")
    ax.axvline(0, color="k", lw=0.8); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
