"""CLI: python -m tyresim [generate|analyze|train|simulate|all]"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, model, plots
from .config import COMPOUNDS, DEFAULT_TRACK
from .data_gen import generate_race_data
from .physics import lap_time as physics_lap_time
from .simulator import RaceSimulator, generate_strategies

DATA = Path("data/laps.csv")
OUT = Path("outputs")


def cmd_generate(a):
    df = generate_race_data(n_races=a.races, n_drivers=a.drivers, seed=a.seed)
    DATA.parent.mkdir(exist_ok=True)
    df.to_csv(DATA, index=False)
    print(f"Dataset: {len(df):,} vueltas -> {DATA}")


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit("No hay datos. Ejecuta primero: python -m tyresim generate")
    return pd.read_csv(DATA, dtype={"compound": "category"})


def prepared() -> pd.DataFrame:
    return analysis.add_stint_delta(analysis.fuel_correct(analysis.clean_laps(load())))


def cmd_analyze(a):
    raw = load()
    df = prepared()
    print(f"Vueltas limpias: {len(df):,} de {len(raw):,}\n")
    deg = analysis.degradation_by_compound(df)
    print("Degradación por compuesto:\n", deg.drop(columns="coefs").round(3), "\n")
    stints = analysis.stint_summary(df)
    print("Degradación media (s/vuelta) por temperatura de pista:\n",
          analysis.temperature_effect(stints), "\n")
    cliffs = analysis.detect_cliff(df)
    print("Vuelta típica del 'cliff' (mediana):\n",
          cliffs.groupby("compound", observed=True)["cliff_lap"].median(), "\n")
    OUT.mkdir(exist_ok=True)
    plots.plot_degradation(df, deg, OUT / "degradation.png")
    print(f"Gráfica -> {OUT / 'degradation.png'}")


def cmd_train(a):
    df = analysis.clean_laps(load())
    for kind in ("ridge", "gbr"):
        r = model.cross_validate(df, kind)
        print(f"{kind:6s} | MAE {r.mae:.3f}s | RMSE {r.rmse:.3f}s | R² {r.r2:.4f}")
    pipe = model.train(df, a.model)
    ages = np.arange(1, 46)
    curves = pd.concat([pd.DataFrame({
        "compound": c, "tyre_age": ages,
        "pred": model.make_lap_time_fn(pipe)(c, ages, np.full(len(ages), 50.0), 35.0),
        "truth": physics_lap_time(c, ages, 50.0, 35.0)}) for c in COMPOUNDS])
    OUT.mkdir(exist_ok=True)
    plots.plot_model_vs_truth(curves, OUT / "model_vs_truth.png")
    print(f"Gráfica -> {OUT / 'model_vs_truth.png'}")


def cmd_simulate(a):
    if a.use_ml:
        pipe = model.train(analysis.clean_laps(load()), a.model)
        sim = RaceSimulator(model.make_lap_time_fn(pipe))
        source = f"modelo ML ({a.model})"
    else:
        sim = RaceSimulator()
        source = "modelo físico"
    strategies = generate_strategies(DEFAULT_TRACK.total_laps, max_stops=a.max_stops)
    print(f"Evaluando {len(strategies):,} estrategias con {source}, pista a {a.temp} °C...\n")
    ranking = sim.rank_strategies(strategies, a.temp, top_k_mc=a.top, n_sims=a.sims)
    cols = ["label", "stops", "time_det", "mc_mean", "mc_std", "mc_p90", "gap", "win_pct"]
    print(ranking[cols].round(2).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    plots.plot_strategies(ranking, OUT / "strategies.png")
    print(f"\nGráfica -> {OUT / 'strategies.png'}")


def main():
    p = argparse.ArgumentParser(prog="tyresim", description="Degradación de neumáticos y simulación de carrera")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--races", type=int, default=20)
    g.add_argument("--drivers", type=int, default=20); g.add_argument("--seed", type=int, default=42)
    sub.add_parser("analyze")
    t = sub.add_parser("train"); t.add_argument("--model", choices=["ridge", "gbr"], default="gbr")
    s = sub.add_parser("simulate")
    s.add_argument("--temp", type=float, default=35.0); s.add_argument("--max-stops", type=int, default=2)
    s.add_argument("--top", type=int, default=8); s.add_argument("--sims", type=int, default=2000)
    s.add_argument("--use-ml", action="store_true"); s.add_argument("--model", choices=["ridge", "gbr"], default="gbr")
    a = p.parse_args()
    {"generate": cmd_generate, "analyze": cmd_analyze, "train": cmd_train, "simulate": cmd_simulate}[a.cmd](a)


if __name__ == "__main__":
    main()
