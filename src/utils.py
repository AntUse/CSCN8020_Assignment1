"""Utility functions for logging, formatting, and timing experiments."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd


def configure_logger(log_path: str = "logs/sample_execution.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("CSCN8020_Assignment1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


@contextmanager
def timer() -> Iterator[dict]:
    result = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start


def values_to_dataframe(values: np.ndarray, rows: int, cols: int, label: str = "V") -> pd.DataFrame:
    grid = np.round(values.reshape(rows, cols), 3)
    return pd.DataFrame(grid, index=[f"row {i}" for i in range(rows)], columns=[f"col {j}" for j in range(cols)])


def policy_to_dataframe(policy_symbols: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(policy_symbols, index=[f"row {i}" for i in range(policy_symbols.shape[0])], columns=[f"col {j}" for j in range(policy_symbols.shape[1])])
