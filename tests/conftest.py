"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_x():
    """x = (1, 2, 3), reell, Wigner-Ville mit M = 8 und Verzögerungen -1 bis 1: W(1, m) = 4 + 6 cos(2 pi m / 8) = (10; 8,2426; 4; -0,2426; -2; -0,2426; 4; 8,2426), W(0, m) = x0^2 = 1 (nur tau = 0)."""
    return np.array([1.0, 2.0, 3.0])
