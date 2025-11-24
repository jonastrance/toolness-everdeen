#!/usr/bin/env python3
"""Test suite for momentum_map.py performance and correctness."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import momentum_map


class TestMomentumMapPerformance(unittest.TestCase):
    """Test performance and correctness of momentum_map functions."""

    def setUp(self):
        """Set up a temporary data file for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        
        # Patch DATA_FILE to use temp file
        self.patcher = patch.object(momentum_map, 'DATA_FILE', self.temp_path)
        self.patcher.start()
        
        # Initialize with empty data
        self.temp_path.write_text('{"areas": {}}')

    def tearDown(self):
        """Clean up temporary file."""
        self.patcher.stop()
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_apply_decay_idempotent(self):
        """Test that applying decay multiple times in short succession doesn't compound."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Create an area
        area = momentum_map.ensure_area(data, "TestArea")
        area["score"] = 100.0
        area["last_decay_ts"] = momentum_map.fmt_ts(now - timedelta(days=1))
        
        # Apply decay once
        momentum_map.apply_decay(area, now)
        score_after_first = area["score"]
        
        # Apply decay again immediately (should not decay further)
        momentum_map.apply_decay(area, now)
        score_after_second = area["score"]
        
        self.assertAlmostEqual(score_after_first, score_after_second, places=7)

    def test_gather_snapshots_no_redundant_decay(self):
        """Test that gather_snapshots applies decay correctly."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Create areas with known scores
        area1 = momentum_map.ensure_area(data, "Area1")
        area1["score"] = 100.0
        area1["last_decay_ts"] = momentum_map.fmt_ts(now - timedelta(days=1))
        
        area2 = momentum_map.ensure_area(data, "Area2")
        area2["score"] = 80.0
        area2["last_decay_ts"] = momentum_map.fmt_ts(now - timedelta(days=2))
        
        # Gather snapshots (applies decay)
        snapshots = momentum_map.gather_snapshots(data, now)
        
        # Verify snapshots are correct
        self.assertEqual(len(snapshots), 2)
        
        # Check that decay was applied
        self.assertLess(snapshots[0].score, 100.0)
        self.assertLess(snapshots[1].score, 80.0)

    def test_history_trimming(self):
        """Test that history is properly trimmed to HISTORY_LIMIT."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Add many updates to exceed HISTORY_LIMIT
        for i in range(momentum_map.HISTORY_LIMIT + 50):
            momentum_map.update_area(data, "TestArea", 1.0, f"Note {i}", now)
        
        area = data["areas"]["TestArea"]
        self.assertEqual(len(area["history"]), momentum_map.HISTORY_LIMIT)
        
        # Verify the most recent entries are kept
        self.assertEqual(area["history"][-1]["note"], f"Note {momentum_map.HISTORY_LIMIT + 49}")

    def test_compute_trend_efficiency(self):
        """Test that compute_trend works correctly with various history sizes."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Create area with known deltas
        deltas = [2.0, 3.0, -1.0, 4.0, 1.0]
        for delta in deltas:
            momentum_map.update_area(data, "TestArea", delta, None, now)
        
        area = data["areas"]["TestArea"]
        trend = momentum_map.compute_trend(area, lookback=5)
        
        # Should be the average of the last 5 deltas
        expected_trend = sum(deltas) / len(deltas)
        self.assertAlmostEqual(trend, expected_trend, places=5)

    def test_momentum_whisper_with_sparse_data(self):
        """Test that momentum_whisper handles sparse data efficiently."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Add only a few entries (less than threshold)
        for i in range(5):
            momentum_map.update_area(data, "TestArea", 1.0, None, now)
        
        whisper = momentum_map.generate_momentum_whisper(data)
        # Should return None for insufficient data
        self.assertIsNone(whisper)

    def test_weekday_bucketing_efficiency(self):
        """Test that weekday bucketing only processes relevant data."""
        data = momentum_map.load_data()
        base_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)  # Monday
        
        # Add entries across multiple weekdays with varying deltas
        for i in range(15):
            day_offset = i % 7
            timestamp = base_time + timedelta(days=day_offset)
            delta = 3.0 if day_offset == 1 else 0.5  # Tuesday has high momentum
            momentum_map.update_area(data, "TestArea", delta, None, timestamp)
        
        whisper = momentum_map.generate_momentum_whisper(data)
        
        # Should identify Tuesday as the high momentum day
        self.assertIsNotNone(whisper)
        self.assertIn("Tuesday", whisper)

    def test_status_output_no_double_decay(self):
        """Test that rendering status doesn't apply decay multiple times."""
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Create area with score
        area = momentum_map.ensure_area(data, "TestArea")
        area["score"] = 100.0
        area["last_decay_ts"] = momentum_map.fmt_ts(now - timedelta(days=1))
        
        momentum_map.save_data(data)
        
        # Render status
        status_output = momentum_map.render_status(data, now)
        
        # Verify output contains the area
        self.assertIn("TestArea", status_output)
        
        # Check that score was decayed once
        expected_score = 100.0 - momentum_map.DECAY_PER_DAY
        self.assertIn(f"{expected_score:.1f}", status_output)

    def test_parse_quick_update(self):
        """Test quick update parsing."""
        area, delta = momentum_map.parse_quick_update("ProjectX +3.5")
        self.assertEqual(area, "ProjectX")
        self.assertEqual(delta, 3.5)
        
        area, delta = momentum_map.parse_quick_update("Deep Work -2")
        self.assertEqual(area, "Deep Work")
        self.assertEqual(delta, -2.0)

    def test_energy_state_classification(self):
        """Test energy state classification from trends."""
        self.assertEqual(momentum_map.energy_state_from_trend(0.5), "Feeding")
        self.assertEqual(momentum_map.energy_state_from_trend(-0.5), "Draining")
        self.assertEqual(momentum_map.energy_state_from_trend(0.1), "Stable")

    def test_clamp_score(self):
        """Test score clamping."""
        self.assertEqual(momentum_map.clamp_score(150.0), 100.0)
        self.assertEqual(momentum_map.clamp_score(-10.0), 0.0)
        self.assertEqual(momentum_map.clamp_score(50.0), 50.0)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks to ensure optimizations don't regress."""

    def setUp(self):
        """Set up test environment."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()
        
        self.patcher = patch.object(momentum_map, 'DATA_FILE', self.temp_path)
        self.patcher.start()
        
        self.temp_path.write_text('{"areas": {}}')

    def tearDown(self):
        """Clean up."""
        self.patcher.stop()
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_large_history_performance(self):
        """Test performance with large history."""
        import time
        
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Add many updates
        start = time.time()
        for i in range(300):
            momentum_map.update_area(data, "TestArea", 1.0, None, now)
        elapsed = time.time() - start
        
        # Should complete reasonably fast
        self.assertLess(elapsed, 2.0, "Adding 300 updates took too long")
        
        # Verify history is trimmed
        area = data["areas"]["TestArea"]
        self.assertEqual(len(area["history"]), momentum_map.HISTORY_LIMIT)

    def test_multiple_areas_performance(self):
        """Test performance with many areas."""
        import time
        
        data = momentum_map.load_data()
        now = datetime.now(timezone.utc)
        
        # Create many areas with history
        start = time.time()
        for i in range(50):
            for j in range(10):
                momentum_map.update_area(data, f"Area{i}", 1.0, None, now)
        elapsed = time.time() - start
        
        # Should handle many areas efficiently
        self.assertLess(elapsed, 3.0, "Creating 50 areas with 10 updates each took too long")
        
        # Gather snapshots
        start = time.time()
        snapshots = momentum_map.gather_snapshots(data, now)
        elapsed = time.time() - start
        
        self.assertEqual(len(snapshots), 50)
        self.assertLess(elapsed, 1.0, "Gathering snapshots for 50 areas took too long")


if __name__ == '__main__':
    unittest.main()
