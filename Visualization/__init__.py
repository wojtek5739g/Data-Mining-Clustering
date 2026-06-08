"""Visualization utilities for clustering experiment results."""

__all__ = ["ResultsVisualizer"]


def __getattr__(name):
    if name == "ResultsVisualizer":
        from .visualization_module import ResultsVisualizer

        return ResultsVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
