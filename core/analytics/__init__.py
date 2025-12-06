"""
Analytics Module
================
Oil market analytics including curves, spreads, and fundamentals.
"""

from .curves import CurveAnalyzer
from .fundamentals import FundamentalAnalyzer
from .spreads import SpreadAnalyzer

__all__ = ["CurveAnalyzer", "SpreadAnalyzer", "FundamentalAnalyzer"]
