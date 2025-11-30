"""
Analytics Module
================
Oil market analytics including curves, spreads, and fundamentals.
"""

from .curves import CurveAnalyzer
from .spreads import SpreadAnalyzer
from .fundamentals import FundamentalAnalyzer

__all__ = ["CurveAnalyzer", "SpreadAnalyzer", "FundamentalAnalyzer"]
