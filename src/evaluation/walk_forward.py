"""
walk_forward.py
===============
Expanding and sliding window Walk-Forward Cross-Validation for environmental time-series.
Ensures zero future leakage across rolling evaluation folds.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Chronological expanding-window cross-validator.

    Folds are defined such that:
    - Fold 1: Train on [0, t1], Test on [t1, t2]
    - Fold 2: Train on [0, t2], Test on [t2, t3]
    ...
    - Fold K: Train on [0, tK], Test on [tK, t_{end}]
    """

    def __init__(
        self,
        n_splits: int = 4,
        test_size_hours: int = 720,  # ~30 days per fold
        min_train_hours: int = 4320,  # ~6 months minimum initial training
    ):
        self.n_splits = n_splits
        self.test_size_hours = test_size_hours
        self.min_train_hours = min_train_hours

    def split(
        self, df: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
        """
        Yields (train_slice, test_slice, fold_idx).
        """
        n = len(df)
        total_eval_hours = self.n_splits * self.test_size_hours
        available_train = n - total_eval_hours

        if available_train < self.min_train_hours:
            raise ValueError(
                f"Dataset length {n} is too short for {self.n_splits} folds with test size {self.test_size_hours}."
            )

        start_test_idx = available_train

        for fold in range(self.n_splits):
            train_end = start_test_idx + fold * self.test_size_hours
            test_end = train_end + self.test_size_hours

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            logger.info(
                "Fold %d: Train [%s to %s] (%d hrs) | Test [%s to %s] (%d hrs)",
                fold + 1,
                train_df.index.min(),
                train_df.index.max(),
                len(train_df),
                test_df.index.min(),
                test_df.index.max(),
                len(test_df),
            )

            yield train_df, test_df, fold + 1
