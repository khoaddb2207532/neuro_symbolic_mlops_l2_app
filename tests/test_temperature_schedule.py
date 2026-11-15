import math

import pytest

from src.rules.temperature import geometric_temperature, penalty_phase


def test_geometric_temperature_warmup_anneal_and_hold():
    values = [
        geometric_temperature(
            epoch=epoch,
            initial_temp=2.0,
            final_temp=12.0,
            warmup_epochs=2,
            anneal_epochs=10,
        )
        for epoch in range(20)
    ]

    assert values[0] == pytest.approx(2.0)
    assert values[1] == pytest.approx(2.0)
    assert values[2] == pytest.approx(2.0)
    assert values[11] == pytest.approx(12.0)
    assert values[19] == pytest.approx(12.0)
    assert all(left <= right for left, right in zip(values, values[1:]))


def test_geometric_temperature_uses_multiplicative_progress():
    midpoint = geometric_temperature(
        epoch=2,
        initial_temp=2.0,
        final_temp=8.0,
        warmup_epochs=0,
        anneal_epochs=5,
    )

    assert midpoint == pytest.approx(math.sqrt(2.0 * 8.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(initial_temp=0.0, final_temp=12.0, warmup_epochs=2, anneal_epochs=10),
        dict(initial_temp=2.0, final_temp=0.0, warmup_epochs=2, anneal_epochs=10),
        dict(initial_temp=2.0, final_temp=12.0, warmup_epochs=-1, anneal_epochs=10),
        dict(initial_temp=2.0, final_temp=12.0, warmup_epochs=2, anneal_epochs=0),
    ],
)
def test_geometric_temperature_rejects_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        geometric_temperature(epoch=0, **kwargs)


def test_penalty_phase_off_then_on_with_local_epoch():
    phases = [penalty_phase(epoch, off_epochs=3) for epoch in range(6)]

    assert phases[:3] == [(False, 0), (False, 0), (False, 0)]
    assert phases[3:] == [(True, 0), (True, 1), (True, 2)]


def test_penalty_phase_zero_off_epochs_is_always_active():
    assert [penalty_phase(e, 0) for e in range(3)] == [(True, 0), (True, 1), (True, 2)]


def test_penalty_phase_restarts_temperature_at_initial_value():
    _, local = penalty_phase(5, off_epochs=5)
    temp = geometric_temperature(
        epoch=local, initial_temp=6.0, final_temp=12.0, warmup_epochs=2, anneal_epochs=10
    )

    assert temp == pytest.approx(6.0)


@pytest.mark.parametrize("epoch,off", [(-1, 0), (0, -1)])
def test_penalty_phase_rejects_negative(epoch: int, off: int):
    with pytest.raises(ValueError):
        penalty_phase(epoch, off)
