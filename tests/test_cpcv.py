"""
Unit tests for Combinatorial Purged Cross-Validation (validator.py).

Verify: no train index in embargo zone; no train sample event window overlapping test.
"""

import numpy as np
import pytest

from validation.validator import (
    CPCVConfig,
    CPCVSplit,
    generate_cpcv_splits,
    write_splits_metadata,
)


class TestCPCVSplitsStructure:
    def test_n_choose_k_splits(self):
        config = CPCVConfig(n_folds=5, k_test=2, horizon_bars=10, embargo_bars=5)
        splits = generate_cpcv_splits(config, n_samples=500)
        assert len(splits) == 10

    def test_train_test_disjoint(self):
        config = CPCVConfig(n_folds=4, k_test=2, horizon_bars=5, embargo_bars=3)
        splits = generate_cpcv_splits(config, n_samples=200)
        for s in splits:
            train_set = set(s.train_idx.tolist())
            test_set = set(s.test_idx.tolist())
            assert len(train_set & test_set) == 0

    def test_no_train_in_embargo_zone(self):
        n_samples = 400
        embargo_bars = 10
        config = CPCVConfig(n_folds=5, k_test=2, horizon_bars=5, embargo_bars=embargo_bars)
        splits = generate_cpcv_splits(config, n_samples)
        for s in splits:
            test_idx = s.test_idx
            test_min, test_max = int(test_idx.min()), int(test_idx.max())
            train_list = s.train_idx.tolist()
            for t in train_list:
                assert not (test_max < t <= test_max + embargo_bars), (
                    f"Train index {t} inside embargo after test end {test_max}"
                )

    def test_no_train_event_overlap_with_test(self):
        n_samples = 300
        horizon_bars = 15
        config = CPCVConfig(n_folds=4, k_test=2, horizon_bars=horizon_bars, embargo_bars=5)
        splits = generate_cpcv_splits(config, n_samples)
        for s in splits:
            test_set = set(s.test_idx.tolist())
            for t in s.train_idx:
                t = int(t)
                event_end = min(t + horizon_bars, n_samples)
                for ti in range(t, event_end):
                    assert ti not in test_set, (
                        f"Train t={t} event [t, t+{horizon_bars}] overlaps test at index {ti}"
                    )


class TestCPCVEdgeCases:
    def test_k_test_equals_n_folds(self):
        config = CPCVConfig(n_folds=3, k_test=3, horizon_bars=2, embargo_bars=2)
        splits = generate_cpcv_splits(config, n_samples=90)
        assert len(splits) <= 1
        if splits:
            assert len(splits[0].test_idx) == 90

    def test_small_n_samples(self):
        config = CPCVConfig(n_folds=3, k_test=2, horizon_bars=2, embargo_bars=1)
        splits = generate_cpcv_splits(config, n_samples=30)
        assert len(splits) == 3


class TestWriteSplitsMetadata:
    def test_writes_json(self, tmp_path):
        config = CPCVConfig(n_folds=3, k_test=2, horizon_bars=2, embargo_bars=1)
        splits = generate_cpcv_splits(config, n_samples=60)
        path = tmp_path / "splits.json"
        write_splits_metadata(path, splits)
        assert path.is_file()
        import json
        data = json.loads(path.read_text())
        assert len(data) == len(splits)
