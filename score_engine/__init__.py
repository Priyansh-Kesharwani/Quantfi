"""
Score Engine - Phase 2

Composite score computation with .fit() and .transform() interface.

The CompositeScorer class provides:
- Configuration-driven scoring
- Batch processing of historical data
- Debug mode for component breakdown
- Integration with indicator engine

Author: Phase 2 Implementation
Date: 2026-02-07
"""

from .scorer import CompositeScorer, ScorerConfig, ScorerResult
from .runner import ScoreRunner, BatchResult

__all__ = [
    'CompositeScorer',
    'ScorerConfig', 
    'ScorerResult',
    'ScoreRunner',
    'BatchResult'
]
