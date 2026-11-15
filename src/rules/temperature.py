# pyright: strict
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


def penalty_phase(epoch: int, off_epochs: int) -> tuple[bool, int]:
    """Cho biết rule penalty có bật ở ``epoch`` không, và epoch cục bộ của lịch nhiệt độ.

    ``off_epochs`` epoch đầu tắt hẳn rule penalty (không khớp luật mềm, loss chỉ còn
    CE + concept BCE). Từ epoch ``off_epochs`` trở đi penalty bật và lịch nhiệt độ
    được tính lại từ 0, để T khởi đầu đúng ``initial_temp`` ở epoch penalty đầu tiên
    thay vì bị "tiêu" trong lúc penalty còn tắt.

    Returns:
        (penalty_active, local_epoch). ``local_epoch`` là 0 khi penalty chưa bật.
    """
    if off_epochs < 0:
        raise ValueError("off_epochs không được âm.")
    if epoch < 0:
        raise ValueError("epoch không được âm.")
    if epoch < off_epochs:
        return False, 0
    return True, epoch - off_epochs
