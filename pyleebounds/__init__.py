"""
pyleebounds: Python package for Lee 2009 treatment effect bounds under sample selection.

This package implements the method from Lee (2009) for estimating sharp bounds
on treatment effects when selection into the post-treatment sample is endogenous.
"""

__version__ = "0.1.0"
__author__ = "Vasco Yasenov"
__email__ = ""

from .lee_bounds import LeeBounds

__all__ = ["LeeBounds"] 