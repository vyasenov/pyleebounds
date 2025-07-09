"""
Implementation of Lee (2009) treatment effect bounds under sample selection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


class LeeBounds:
    """
    Lee (2009) treatment effect bounds estimator.
    
    Implements the method from Lee (2009) for estimating sharp bounds on treatment
    effects when selection into the post-treatment sample is endogenous.
    
    Parameters
    ----------
    None
    
    References
    ----------
    Lee, D. S. (2009). Training, wages, and sample selection: Estimating sharp 
    bounds on treatment effects. The Review of Economic Studies, 76(3), 1071-1102.
    """
    
    def __init__(self, trim_method: str = 'quantile'):
        self.trim_method = trim_method
        self.fitted = False
        self.results = None
        
    def fit(self, data: pd.DataFrame, outcome: str, treatment: str, 
            selection: str) -> 'LeeBoundsResults':
        """
        Fit Lee bounds estimator to the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data containing outcome, treatment, and selection variables
        outcome : str
            Name of the outcome variable column
        treatment : str
            Name of the treatment indicator column (0=control, 1=treated)
        selection : str
            Name of the selection indicator column (0=missing, 1=observed)
            
        Returns
        -------
        LeeBoundsResults
            Results object containing bounds and summary statistics
        """
        # Validate inputs
        required_cols = [outcome, treatment, selection]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Extract data
        Y = data[outcome].values
        D = data[treatment].values
        S = data[selection].values
        
        # Remove missing values
        valid_mask = ~(np.isnan(Y) | np.isnan(D) | np.isnan(S))
        Y, D, S = Y[valid_mask], D[valid_mask], S[valid_mask]
        
        # Validate data types and values
        self._validate_variables(Y, D, S)
        
        # Calculate selection rates
        p1 = np.mean(S[D == 1])  # Selection rate in treated group
        p0 = np.mean(S[D == 0])  # Selection rate in control group
        
        if p1 <= p0:
            raise ValueError("Selection rate in treated group must be greater than control group")
        
        # Calculate bounds
        lower_bound, upper_bound = self._compute_bounds(Y, D, S, p1, p0)
        
        # Store results
        self.results = LeeBoundsResults(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            p1=p1,
            p0=p0,
            trim_proportion=p1 - p0,
            n_treated=np.sum(D == 1),
            n_control=np.sum(D == 0),
            n_treated_selected=np.sum((D == 1) & (S == 1)),
            n_control_selected=np.sum((D == 0) & (S == 1))
        )
        
        self.fitted = True
        return self.results
    
    def _compute_bounds(self, Y: np.ndarray, D: np.ndarray, S: np.ndarray, 
                       p1: float, p0: float) -> Tuple[float, float]:
        """
        Compute Lee bounds using trimming approach.
        
        Parameters
        ----------
        Y : np.ndarray
            Outcome values
        D : np.ndarray
            Treatment indicators
        S : np.ndarray
            Selection indicators
        p1 : float
            Selection rate in treated group
        p0 : float
            Selection rate in control group
            
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        # Control group mean (among selected)
        control_mean = np.mean(Y[(D == 0) & (S == 1)])
        
        # Treated group (among selected)
        treated_selected = Y[(D == 1) & (S == 1)]
        
        # Trim proportion
        trim_prop = p1 - p0
        
        # For Lee bounds, we trim the treated group to match control selection rate
        # Lower bound: trim from top (keep lowest outcomes)
        # Upper bound: trim from bottom (keep highest outcomes)
        n_trim = int(len(treated_selected) * trim_prop)
        
        if n_trim > 0:
            sorted_treated = np.sort(treated_selected)
            # Lower bound: keep bottom (1 - trim_prop) of observations
            lower_trimmed_mean = np.mean(sorted_treated[:-n_trim])
            # Upper bound: keep top (1 - trim_prop) of observations  
            upper_trimmed_mean = np.mean(sorted_treated[n_trim:])
        else:
            lower_trimmed_mean = upper_trimmed_mean = np.mean(treated_selected)
        
        lower_bound = lower_trimmed_mean - control_mean
        upper_bound = upper_trimmed_mean - control_mean
        
        return lower_bound, upper_bound
    
    def _validate_variables(self, Y: np.ndarray, D: np.ndarray, S: np.ndarray) -> None:
        """
        Validate that variables have correct types and values.
        
        Parameters
        ----------
        Y : np.ndarray
            Outcome variable
        D : np.ndarray
            Treatment variable
        S : np.ndarray
            Selection variable
            
        Raises
        ------
        ValueError
            If validation fails
        """
        # Check that D is binary with values 0 and 1
        unique_d = np.unique(D)
        if not np.array_equal(unique_d, np.array([0, 1])):
            raise ValueError(f"Treatment variable D must be binary (0, 1). Found values: {unique_d}")
        
        # Check that S is binary with values 0 and 1
        unique_s = np.unique(S)
        if not np.array_equal(unique_s, np.array([0, 1])):
            raise ValueError(f"Selection variable S must be binary (0, 1). Found values: {unique_s}")
        
        # Check that Y is continuous (not all integers)
        if len(np.unique(Y)) < len(Y) * 0.1:  # If less than 10% unique values, likely discrete
            raise ValueError("Outcome variable Y should be continuous. Consider if this is appropriate.")
        
        # Check for reasonable sample sizes
        if len(Y) < 10:
            raise ValueError("Sample size too small. Need at least 10 observations.")
        
        # Check that we have both treatment groups
        if np.sum(D == 0) == 0:
            raise ValueError("No control observations (D=0) found.")
        if np.sum(D == 1) == 0:
            raise ValueError("No treated observations (D=1) found.")
        
        # Check that we have selected observations in both groups
        if np.sum((D == 0) & (S == 1)) == 0:
            raise ValueError("No selected control observations (D=0, S=1) found.")
        if np.sum((D == 1) & (S == 1)) == 0:
            raise ValueError("No selected treated observations (D=1, S=1) found.")

    def bootstrap(self, data: pd.DataFrame, outcome: str, treatment: str, 
                  selection: str, n_bootstrap: int = 500, 
                  ci_level: float = 0.95) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals for the bounds.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome : str
            Outcome variable name
        treatment : str
            Treatment variable name
        selection : str
            Selection variable name
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level (e.g., 0.95 for 95% CI)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing bootstrap results
        """
        lower_bounds = []
        upper_bounds = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[boot_idx].reset_index(drop=True)
            
            try:
                # Fit Lee bounds on bootstrap sample
                lb = LeeBounds()
                results = lb.fit(boot_data, outcome, treatment, selection)
                lower_bounds.append(results.lower_bound)
                upper_bounds.append(results.upper_bound)
            except:
                # Skip if bootstrap sample fails
                continue
        
        # Calculate confidence intervals
        alpha = 1 - ci_level
        lower_ci = np.percentile(lower_bounds, [alpha/2*100, (1-alpha/2)*100])
        upper_ci = np.percentile(upper_bounds, [alpha/2*100, (1-alpha/2)*100])
        
        return {
            'lower_bound_ci': lower_ci,
            'upper_bound_ci': upper_ci,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'ci_level': ci_level
        } 


class LeeBoundsResults:
    """
    Results from Lee bounds estimation.
    """
    
    def __init__(self, lower_bound: float, upper_bound: float, p1: float, 
                 p0: float, trim_proportion: float, n_treated: int, n_control: int,
                 n_treated_selected: int, n_control_selected: int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p1 = p1
        self.p0 = p0
        self.trim_proportion = trim_proportion
        self.n_treated = n_treated
        self.n_control = n_control
        self.n_treated_selected = n_treated_selected
        self.n_control_selected = n_control_selected
        
    def summary(self) -> str:
        """
        Return a summary of the results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        summary = f"""
Lee (2009) Treatment Effect Bounds
=================================

Treatment Effect Bounds:
- Lower bound: {self.lower_bound:.4f}
- Upper bound: {self.upper_bound:.4f}
- Bound width: {self.upper_bound - self.lower_bound:.4f}
"""
        return summary