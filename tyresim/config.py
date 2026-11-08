"""Parámetros del circuito y de los compuestos de neumático.

Todos los valores son ficticios pero con órdenes de magnitud realistas
(tiempos en segundos, combustible en kg).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Compound:
    name: str
    pace_offset: float   # s/vuelta más lento que el compuesto más rápido (neumático nuevo)
    deg_linear: float    # s/vuelta perdidos por cada vuelta de uso
    deg_quad: float      # término cuadrático (la degradación se acelera)
    cliff_lap: int       # vuelta de vida a partir de la cual el neumático "se cae"
    cliff_rate: float    # s/vuelta extra por cada vuelta pasada del cliff


COMPOUNDS: dict[str, Compound] = {
    "SOFT":   Compound("SOFT",   0.00, 0.080, 0.0025, 18, 0.30),
    "MEDIUM": Compound("MEDIUM", 0.55, 0.050, 0.0012, 28, 0.25),
    "HARD":   Compound("HARD",   1.05, 0.030, 0.0006, 40, 0.20),
}


@dataclass(frozen=True)
class Track:
    name: str = "Circuito Demo"
    total_laps: int = 57
    base_lap_time: float = 90.0     # s, coche vacío con neumático blando nuevo
    pit_loss: float = 22.0          # s perdidos al entrar a boxes
    fuel_start: float = 110.0       # kg
    fuel_effect: float = 0.035      # s por kg de combustible
    ref_track_temp: float = 35.0    # °C de referencia
    temp_deg_factor: float = 0.025  # +2.5 % de degradación por cada °C sobre la referencia
    lap_noise_std: float = 0.25     # ruido de vuelta (s)
    sc_probability: float = 0.012   # prob. de safety car en cada vuelta
    sc_duration: int = 4            # vueltas que dura un safety car
    sc_lap_factor: float = 1.40     # vueltas bajo SC son 40 % más lentas
    sc_pit_discount: float = 0.5    # bajo SC la parada cuesta la mitad

    @property
    def fuel_per_lap(self) -> float:
        return self.fuel_start / self.total_laps


DEFAULT_TRACK = Track()
