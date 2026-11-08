"""Modelo "real" (ground truth) del tiempo por vuelta.

Se usa para generar los datos sintéticos y como referencia para comparar
contra el modelo de Machine Learning.
"""
import numpy as np

from .config import COMPOUNDS, DEFAULT_TRACK, Track


def tyre_degradation(compound: str, tyre_age, track_temp, track: Track = DEFAULT_TRACK):
    """Segundos perdidos por desgaste. Vectorizado: acepta arrays de NumPy."""
    c = COMPOUNDS[compound]
    age = np.asarray(tyre_age, dtype=float)
    temp_mult = 1.0 + track.temp_deg_factor * (np.asarray(track_temp, dtype=float) - track.ref_track_temp)
    cliff = np.clip(age - c.cliff_lap, 0, None) * c.cliff_rate
    return (c.deg_linear * age + c.deg_quad * age**2 + cliff) * temp_mult


def lap_time(compound: str, tyre_age, fuel_kg, track_temp, track: Track = DEFAULT_TRACK):
    """Tiempo de vuelta esperado (sin ruido)."""
    c = COMPOUNDS[compound]
    return (
        track.base_lap_time
        + c.pace_offset
        + track.fuel_effect * np.asarray(fuel_kg, dtype=float)
        + tyre_degradation(compound, tyre_age, track_temp, track)
    )
