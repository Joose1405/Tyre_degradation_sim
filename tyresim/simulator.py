"""Simulador de carrera y optimizador de estrategias (Monte Carlo con NumPy)."""
from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from .config import COMPOUNDS, DEFAULT_TRACK, Track
from .physics import lap_time as physics_lap_time

LapTimeFn = Callable[[str, np.ndarray, np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class Strategy:
    stints: tuple[tuple[str, int], ...]   # ((compuesto, vueltas), ...)

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def pit_laps(self) -> list[int]:
        return np.cumsum([n for _, n in self.stints])[:-1].tolist()

    def __str__(self) -> str:
        return " -> ".join(f"{c[0]}{n}" for c, n in self.stints)


def generate_strategies(total_laps: int, max_stops: int = 2, min_stint: int = 8,
                        step: int = 2) -> list[Strategy]:
    """Enumera estrategias válidas (reglamento: al menos 2 compuestos distintos)."""
    strategies = []
    names = list(COMPOUNDS)
    for stops in range(1, max_stops + 1):
        n = stops + 1
        for compounds in product(names, repeat=n):
            if len(set(compounds)) < 2:
                continue
            for cuts in product(range(min_stint, total_laps - min_stint + 1, step), repeat=stops):
                if list(cuts) != sorted(set(cuts)):
                    continue
                lengths = np.diff([0, *cuts, total_laps])
                if lengths.min() >= min_stint:
                    strategies.append(Strategy(tuple(zip(compounds, map(int, lengths)))))
    return strategies


class RaceSimulator:
    def __init__(self, lap_time_fn: LapTimeFn | None = None, track: Track = DEFAULT_TRACK):
        self.track = track
        self.lap_time_fn = lap_time_fn or (lambda c, a, f, t: physics_lap_time(c, a, f, t, track))
        self._cache: dict = {}

    def _grid(self, compound: str, track_temp: float) -> np.ndarray:
        """Matriz G[vuelta_inicio, edad-1] con el tiempo de vuelta para un compuesto.

        Se predice toda la malla de una vez (una sola llamada al modelo por compuesto),
        lo que hace viable evaluar miles de estrategias con un modelo de ML.
        """
        key = (compound, round(track_temp, 2))
        if key not in self._cache:
            tr, n = self.track, self.track.total_laps
            start, age = np.meshgrid(np.arange(n), np.arange(1, n + 1), indexing="ij")
            fuel = np.clip(tr.fuel_start - tr.fuel_per_lap * (start + age - 1), 0, None)
            pred = self.lap_time_fn(compound, age.ravel(), fuel.ravel(), track_temp)
            self._cache[key] = np.asarray(pred).reshape(n, n)
        return self._cache[key]

    def base_lap_times(self, strategy: Strategy, track_temp: float) -> np.ndarray:
        """Tiempos de vuelta deterministas (sin ruido ni safety car) de toda la carrera."""
        times, lap = [], 0
        for compound, length in strategy.stints:
            times.append(self._grid(compound, track_temp)[lap, :length])
            lap += length
        return np.concatenate(times)

    def race_time(self, strategy: Strategy, track_temp: float = 35.0) -> float:
        return float(self.base_lap_times(strategy, track_temp).sum() + strategy.n_stops * self.track.pit_loss)

    def monte_carlo(self, strategy: Strategy, track_temp: float = 35.0, n_sims: int = 2000,
                    seed: int | None = 0) -> np.ndarray:
        """Devuelve `n_sims` tiempos totales con ruido por vuelta y safety cars aleatorios.

        Todo vectorizado: matrices de forma (n_sims, total_laps).
        """
        tr, rng = self.track, np.random.default_rng(seed)
        laps = tr.total_laps
        base = self.base_lap_times(strategy, track_temp)
        times = base + rng.normal(0, tr.lap_noise_std, size=(n_sims, laps))

        # Safety car: arranca con prob. p en cada vuelta y dura `sc_duration` vueltas
        starts = rng.random((n_sims, laps)) < tr.sc_probability
        kernel = np.ones(tr.sc_duration)
        under_sc = np.apply_along_axis(lambda r: np.convolve(r, kernel)[:laps] > 0, 1, starts)
        sc_time = base.mean() * tr.sc_lap_factor
        times = np.where(under_sc, np.maximum(times, sc_time), times)

        # Parar bajo SC sale más barato
        pit_idx = np.array(strategy.pit_laps, dtype=int) - 1
        pit_cost = np.full((n_sims, len(pit_idx)), tr.pit_loss)
        if len(pit_idx):
            pit_cost[under_sc[:, pit_idx]] *= tr.sc_pit_discount
        return times.sum(axis=1) + pit_cost.sum(axis=1)

    def rank_strategies(self, strategies: list[Strategy], track_temp: float = 35.0,
                        top_k_mc: int = 10, n_sims: int = 2000, seed: int = 0) -> pd.DataFrame:
        """1) Ranking determinista de todas. 2) Monte Carlo sobre las `top_k_mc` mejores.

        Todas las estrategias usan la MISMA semilla ("common random numbers"): cada simulación i
        tiene los mismos safety cars y el mismo ruido para todas, así las diferencias reflejan
        la estrategia y no la suerte.
        """
        det = pd.DataFrame({"strategy": strategies,
                            "stops": [s.n_stops for s in strategies],
                            "time_det": [self.race_time(s, track_temp) for s in strategies]})
        # Mejor variante de cada combinación de compuestos para tener diversidad
        det["compounds"] = det["strategy"].map(lambda s: "-".join(c for c, _ in s.stints))
        top = det.sort_values("time_det").drop_duplicates("compounds").head(top_k_mc).copy()
        mc = [self.monte_carlo(s, track_temp, n_sims, seed=seed) for s in top["strategy"]]
        top["mc_mean"] = [m.mean() for m in mc]
        top["mc_std"] = [m.std() for m in mc]
        top["mc_p90"] = [np.percentile(m, 90) for m in mc]
        top["gap"] = top["mc_mean"] - top["mc_mean"].min()
        # % de simulaciones en las que cada estrategia es la más rápida
        wins = np.argmin(np.vstack(mc), axis=0)
        top["win_pct"] = np.bincount(wins, minlength=len(mc)) / n_sims * 100
        top["label"] = top["strategy"].map(str)
        top["_samples"] = mc
        return top.sort_values("mc_mean").reset_index(drop=True)
