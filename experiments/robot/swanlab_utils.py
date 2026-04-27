"""Shared SwanLab helpers for robot evaluation scripts."""

from pathlib import Path

try:
    import swanlab
except ImportError:
    swanlab = None


def _sanitize_config(config):
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def maybe_init_swanlab(enabled, project, experiment_name, config):
    """Initialize SwanLab only when requested."""
    if not enabled:
        return False
    if swanlab is None:
        raise ImportError("SwanLab logging requested, but `swanlab` is not installed in the current environment.")

    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config=_sanitize_config(config),
    )
    return True


def maybe_log_swanlab(enabled, metrics, step=None):
    """Log metrics to SwanLab only when initialized."""
    if not enabled:
        return
    if step is None:
        swanlab.log(metrics)
    else:
        swanlab.log(metrics, step=step)
