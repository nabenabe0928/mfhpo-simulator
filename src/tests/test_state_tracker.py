from __future__ import annotations

import unittest

import numpy as np

from src._constants import _StateType
from src._state_tracker import _AskTellStateTracker


def test_fetch_prev_seed_no_prior_state():
    """When no prior state exists, fetch_prev_seed should return a new random seed."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    rng = np.random.RandomState(42)
    seed = tracker.fetch_prev_seed(config_hash=0, fidel=3, cumtime=0.0, rng=rng)
    assert isinstance(seed, (int, np.integer))
    assert len(tracker._intermediate_states) == 0


def test_fetch_prev_seed_with_prior_state():
    """When a prior state exists at a lower fidelity, fetch_prev_seed should return the cached seed."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=3, runtime=100.0, cumtime=50.0, seed=999)

    rng = np.random.RandomState(42)
    seed = tracker.fetch_prev_seed(config_hash=0, fidel=5, cumtime=100.0, rng=rng)
    assert seed == 999


def test_fetch_prev_seed_ignores_future_cumtime():
    """Prior state with cumtime > current cumtime should not be found."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=3, runtime=100.0, cumtime=200.0, seed=999)

    rng = np.random.RandomState(42)
    seed = tracker.fetch_prev_seed(config_hash=0, fidel=5, cumtime=100.0, rng=rng)
    # Should get a new random seed, not the cached one
    assert seed != 999 or seed == rng.randint(1 << 30)  # just check it's from rng


def test_fetch_prev_seed_ignores_higher_fidel():
    """Prior state with fidel >= current fidel should not be found."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=5, runtime=100.0, cumtime=50.0, seed=999)

    rng = np.random.RandomState(42)
    seed = tracker.fetch_prev_seed(config_hash=0, fidel=5, cumtime=100.0, rng=rng)
    # fidel=5 is not < 5, so should get new random seed
    assert seed != 999


def test_pop_old_state_returns_none_when_no_prior():
    """pop_old_state should return None when no matching prior state exists."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    result = tracker.pop_old_state(config_hash=0, fidel=3, cumtime=100.0)
    assert result is None


def test_pop_old_state_returns_and_removes_state():
    """pop_old_state should return the matching state and remove it from cache."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=3, runtime=100.0, cumtime=50.0, seed=42)
    assert 0 in tracker._intermediate_states

    old_state = tracker.pop_old_state(config_hash=0, fidel=5, cumtime=100.0)
    assert old_state is not None
    assert old_state.fidel == 3
    assert old_state.seed == 42
    assert old_state.runtime == 100.0
    # Should have cleaned up the empty entry
    assert 0 not in tracker._intermediate_states


def test_pop_old_state_keeps_other_states():
    """pop_old_state should only remove the matching state, keeping others for the same config."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=2, runtime=50.0, cumtime=30.0, seed=11)
    tracker.update_state(config_hash=0, fidel=5, runtime=150.0, cumtime=80.0, seed=22)

    # Pop the one at fidel=2 (both have fidel < 7, but index(True) returns the first match)
    old_state = tracker.pop_old_state(config_hash=0, fidel=7, cumtime=100.0)
    assert old_state is not None
    assert old_state.fidel == 2
    # The fidel=5 state should still be there
    assert 0 in tracker._intermediate_states
    assert len(tracker._intermediate_states[0]) == 1
    assert tracker._intermediate_states[0][0].fidel == 5


def test_update_state_does_not_cache_at_max_fidel():
    """update_state should NOT cache when fidel == continual_max_fidel."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=10, runtime=100.0, cumtime=50.0, seed=42)
    assert len(tracker._intermediate_states) == 0


def test_update_state_caches_below_max_fidel():
    """update_state should cache when fidel < continual_max_fidel."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=5, runtime=100.0, cumtime=50.0, seed=42)
    assert 0 in tracker._intermediate_states
    assert len(tracker._intermediate_states[0]) == 1
    state = tracker._intermediate_states[0][0]
    assert state == _StateType(runtime=100.0, cumtime=50.0, fidel=5, seed=42)


def test_update_state_appends_to_existing_config():
    """update_state should append new states for the same config_hash."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=3, runtime=50.0, cumtime=30.0, seed=11)
    tracker.update_state(config_hash=0, fidel=5, runtime=100.0, cumtime=60.0, seed=22)
    assert len(tracker._intermediate_states[0]) == 2


def test_multiple_configs_independent():
    """Different config_hash values should be tracked independently."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=3, runtime=50.0, cumtime=30.0, seed=11)
    tracker.update_state(config_hash=1, fidel=5, runtime=100.0, cumtime=60.0, seed=22)

    assert len(tracker._intermediate_states) == 2
    assert len(tracker._intermediate_states[0]) == 1
    assert len(tracker._intermediate_states[1]) == 1

    # Popping from config 0 should not affect config 1
    tracker.pop_old_state(config_hash=0, fidel=5, cumtime=100.0)
    assert 0 not in tracker._intermediate_states
    assert 1 in tracker._intermediate_states


def test_fetch_prev_state_index_returns_first_match():
    """_fetch_prev_state_index returns index(True) — the first matching state."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=2, runtime=50.0, cumtime=10.0, seed=11)
    tracker.update_state(config_hash=0, fidel=4, runtime=100.0, cumtime=20.0, seed=22)

    # Both fidel=2 and fidel=4 are < 6, and both cumtime <= 30; first match is index 0
    idx = tracker._fetch_prev_state_index(config_hash=0, fidel=6, cumtime=30.0)
    assert idx == 0


def test_fetch_prev_state_index_no_match():
    """_fetch_prev_state_index returns None when no state matches."""
    tracker = _AskTellStateTracker(continual_max_fidel=10)
    tracker.update_state(config_hash=0, fidel=5, runtime=100.0, cumtime=200.0, seed=42)

    # fidel=5 is not < 3, so no match
    idx = tracker._fetch_prev_state_index(config_hash=0, fidel=3, cumtime=300.0)
    assert idx is None

    # cumtime=100 is not <= 50, so no match
    idx = tracker._fetch_prev_state_index(config_hash=0, fidel=8, cumtime=50.0)
    assert idx is None


if __name__ == "__main__":
    unittest.main()
