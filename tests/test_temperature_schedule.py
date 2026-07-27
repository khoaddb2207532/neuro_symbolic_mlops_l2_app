import math

import pytest

from src.rules.temperature import geometric_temperature


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
