import numpy as np
import pytest

from tyresim import analysis, model
from tyresim.config import DEFAULT_TRACK
from tyresim.data_gen import generate_race_data
from tyresim.physics import lap_time
from tyresim.simulator import RaceSimulator, Strategy, generate_strategies


@pytest.fixture(scope="module")
def raw():
    return generate_race_data(n_races=4, n_drivers=10, seed=1)


@pytest.fixture(scope="module")
def prepared(raw):
    return analysis.add_stint_delta(analysis.fuel_correct(analysis.clean_laps(raw)))


def test_dataset_has_every_lap(raw):
    laps_per_driver = raw.groupby(["race_id", "driver"])["lap"].count()
    assert (laps_per_driver == DEFAULT_TRACK.total_laps).all()


def test_clean_removes_pit_laps(raw):
    clean = analysis.clean_laps(raw)
    assert not clean["pit_in"].any() and not clean["pit_out"].any()
    assert len(clean) < len(raw)


def test_soft_degrades_more_than_hard(prepared):
    deg = analysis.degradation_by_compound(prepared)
    assert deg.loc["SOFT", "deg_s_per_lap"] > deg.loc["MEDIUM", "deg_s_per_lap"] > deg.loc["HARD", "deg_s_per_lap"]


def test_physics_hotter_track_degrades_more():
    cool = lap_time("SOFT", 20, 50, 25) - lap_time("SOFT", 1, 50, 25)
    hot = lap_time("SOFT", 20, 50, 45) - lap_time("SOFT", 1, 50, 45)
    assert hot > cool


def test_strategies_are_legal():
    strategies = generate_strategies(57, max_stops=2)
    assert strategies
    for s in strategies:
        assert sum(n for _, n in s.stints) == 57
        assert len({c for c, _ in s.stints}) >= 2


def test_race_time_includes_pit_loss():
    sim = RaceSimulator()
    s = Strategy((("MEDIUM", 30), ("HARD", 27)))
    assert sim.race_time(s) == pytest.approx(sim.base_lap_times(s, 35.0).sum() + DEFAULT_TRACK.pit_loss)


def test_monte_carlo_is_reproducible():
    sim = RaceSimulator()
    s = Strategy((("SOFT", 20), ("HARD", 37)))
    a, b = sim.monte_carlo(s, n_sims=200, seed=7), sim.monte_carlo(s, n_sims=200, seed=7)
    np.testing.assert_allclose(a, b)
    assert a.min() >= sim.base_lap_times(s, 35.0).sum() - 50


def test_ml_model_close_to_physics(raw):
    pipe = model.train(analysis.clean_laps(raw), "ridge")
    fn = model.make_lap_time_fn(pipe)
    ages = np.arange(1, 20)
    pred = fn("MEDIUM", ages, np.full(len(ages), 60.0), 35.0)
    truth = lap_time("MEDIUM", ages, 60.0, 35.0)
    assert np.abs(pred - truth).mean() < 0.5
