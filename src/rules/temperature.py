def geometric_temperature(
    epoch: int,
    initial_temp: float,
    final_temp: float,
    warmup_epochs: int,
    anneal_epochs: int,
) -> float:
    """Warmup -> geometric annealing -> hold, independent of max epochs."""
    if initial_temp <= 0 or final_temp <= 0:
        raise ValueError("initial_temp và final_temp phải lớn hơn 0.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs không được âm.")
    if anneal_epochs < 1:
        raise ValueError("anneal_epochs phải ít nhất là 1.")
    if epoch < warmup_epochs:
        return float(initial_temp)

    progress = (epoch - warmup_epochs) / max(anneal_epochs - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(initial_temp * (final_temp / initial_temp) ** progress)
