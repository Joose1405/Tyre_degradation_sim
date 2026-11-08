"""Generación de un dataset sintético de vueltas de carrera."""
import numpy as np
import pandas as pd

from .config import COMPOUNDS, DEFAULT_TRACK, Track
from .physics import lap_time


def _random_strategy(rng: np.random.Generator, total_laps: int) -> list[tuple[str, int]]:
    """Estrategia aleatoria de 1 o 2 paradas con al menos 2 compuestos distintos."""
    names = list(COMPOUNDS)
    while True:
        n_stints = rng.choice([2, 3], p=[0.6, 0.4])
        cuts = np.sort(rng.choice(np.arange(10, total_laps - 9), size=n_stints - 1, replace=False))
        lengths = np.diff(np.concatenate([[0], cuts, [total_laps]]))
        compounds = rng.choice(names, size=n_stints)
        if len(set(compounds)) >= 2 and lengths.min() >= 8:
            return list(zip(compounds.tolist(), lengths.tolist()))


def generate_race_data(
    n_races: int = 20,
    n_drivers: int = 20,
    seed: int = 42,
    track: Track = DEFAULT_TRACK,
) -> pd.DataFrame:
    """Simula `n_races` carreras con `n_drivers` pilotos y devuelve una fila por vuelta."""
    rng = np.random.default_rng(seed)
    rows = []
    for race_id in range(n_races):
        track_temp = rng.uniform(25, 50)
        for d in range(n_drivers):
            driver = f"DRV{d:02d}"
            skill = rng.normal(0, 0.35)  # ritmo propio del piloto/coche
            lap = 0
            for stint, (compound, length) in enumerate(_random_strategy(rng, track.total_laps), start=1):
                ages = np.arange(1, length + 1)
                laps = lap + ages
                fuel = track.fuel_start - track.fuel_per_lap * (laps - 1)
                times = lap_time(compound, ages, fuel, track_temp, track) + skill
                times += rng.normal(0, track.lap_noise_std, size=length)
                # Tráfico: algunas vueltas pierden 1-3 s
                traffic = rng.random(length) < 0.04
                times[traffic] += rng.uniform(1, 3, traffic.sum())
                for i in range(length):
                    rows.append((race_id, driver, int(laps[i]), stint, compound, int(ages[i]),
                                 float(fuel[i]), track_temp, float(times[i]), bool(traffic[i])))
                lap += length

    df = pd.DataFrame(rows, columns=["race_id", "driver", "lap", "stint", "compound", "tyre_age",
                                     "fuel_kg", "track_temp", "lap_time", "traffic"])
    # Vueltas de entrada / salida de boxes
    last_lap_of_stint = df.groupby(["race_id", "driver", "stint"])["lap"].transform("max")
    df["pit_in"] = (df["lap"] == last_lap_of_stint) & (df["lap"] < track.total_laps)
    df["pit_out"] = (df["tyre_age"] == 1) & (df["stint"] > 1)
    df.loc[df["pit_in"], "lap_time"] += track.pit_loss * 0.4
    df.loc[df["pit_out"], "lap_time"] += track.pit_loss * 0.6
    df["compound"] = df["compound"].astype("category")
    return df
