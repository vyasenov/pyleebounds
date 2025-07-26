"""
Implementation of Lee (2009) treatment effect bounds under sample selection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

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
    
    def __init__(self, n_bootstrap: int = 100, ci_level: float = 0.95):
        """
        Initialize Lee bounds estimator.
        
        Parameters
        ----------
        n_bootstrap : int, default=100
            Number of bootstrap samples for confidence intervals
        ci_level : float, default=0.95
            Confidence level for bootstrap confidence intervals
        """
        self.n_bootstrap = n_bootstrap
        self.ci_level = ci_level
    
    def _compute_bounds(self, Y: np.ndarray, D: np.ndarray, S: np.ndarray, 
                       p1: float, p0: float) -> Tuple[float, float]:
        """
        Compute Lee treatment effect bounds using trimming approach.
        
        Parameters
        ----------
        Y : np.ndarray; Outcome values
        D : np.ndarray; Treatment indicators
        S : np.ndarray; Selection indicators
        p1 : float;  Selection rate in treated group
        p0 : float;  Selection rate in control group
            
        Returns
        -------
        Tuple[float, float]
            (lower_bound, upper_bound)
        """
        # Control group mean (among selected)
        control_mean = np.mean(Y[(D == 0) & (S == 1)])
        
        # Treated group (among selected)
        treated_selected = Y[(D == 1) & (S == 1)]
        
        # For Lee bounds, we trim the treated group to match control selection rate
        # Lower bound: trim from top (keep lowest outcomes)
        # Upper bound: trim from bottom (keep highest outcomes)
        if p1 > p0:
            # Trim proportion
            trim_prop = (p1 - p0) / p1
            n_trim = int(len(treated_selected) * trim_prop)
            
            if n_trim >= len(treated_selected):
                # Handle case where we'd trim everything
                raise ValueError("Trim proportion too large - would remove all observations")

            sorted_treated = np.sort(treated_selected)
            # Lower bound: keep bottom (1 - trim_prop) of observations (trimmed from top)
            lower_trimmed_mean = np.mean(sorted_treated[:-n_trim])
            # Upper bound: keep top (1 - trim_prop) of observations  
            upper_trimmed_mean = np.mean(sorted_treated[n_trim:])
        else:
            lower_trimmed_mean = upper_trimmed_mean = np.mean(treated_selected)
        
        lower_bound = lower_trimmed_mean - control_mean
        upper_bound = upper_trimmed_mean - control_mean
        
        return lower_bound, upper_bound
    
    def _validate_data(self, Y: np.ndarray, D: np.ndarray, S: np.ndarray) -> None:
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

    def _bootstrap(self, data: pd.DataFrame, outcome: str, treatment: str, 
                  selection: str) -> Dict[str, Any]:
        """
        Compute bootstrap confidence intervals for the bounds.
        
        Parameters
        ----------
        data : pd.DataFrame; Input data
        outcome : str; Outcome variable name
        treatment : str; Treatment variable name
        selection : str; Selection variable name
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing bootstrap results
        """
        lower_bounds = []
        upper_bounds = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(len(data), size=len(data), replace=True)
            boot_data = data.iloc[boot_idx].reset_index(drop=True)
            
            try:
                # Extract data from bootstrap sample
                Y = boot_data[outcome].values
                D = boot_data[treatment].values
                S = boot_data[selection].values
                
                # Remove missing values
                valid_mask = ~(np.isnan(Y) | np.isnan(D) | np.isnan(S))
                Y, D, S = Y[valid_mask], D[valid_mask], S[valid_mask]
                
                # Validate data types and values
                self._validate_data(Y, D, S)
                
                # Calculate selection rates
                p1 = np.mean(S[D == 1])
                p0 = np.mean(S[D == 0])
                
                if p1 <= p0:
                    continue  # Skip this bootstrap sample
                
                # Calculate bounds directly
                lower_bound, upper_bound = self._compute_bounds(Y, D, S, p1, p0)
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
                
            except Exception:
                # Skip if bootstrap sample fails
                continue
        
        # Calculate confidence intervals and standard errors
        if len(lower_bounds) > 0:
            alpha = 1 - self.ci_level
            lower_ci = np.percentile(lower_bounds, [alpha/2*100, (1-alpha/2)*100])
            upper_ci = np.percentile(upper_bounds, [alpha/2*100, (1-alpha/2)*100])
            
            # Calculate bootstrap standard errors
            lower_bound_se = np.std(lower_bounds, ddof=1)
            upper_bound_se = np.std(upper_bounds, ddof=1)
        else:
            # If no successful bootstrap samples, use point estimates
            lower_ci = upper_ci = np.array([np.nan, np.nan])
            lower_bound_se = upper_bound_se = np.nan
        
        return {
            'lower_bound_ci': lower_ci,
            'upper_bound_ci': upper_ci,
            'lower_bound_se': lower_bound_se,
            'upper_bound_se': upper_bound_se,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'ci_level': self.ci_level
        }

    def fit(self, data: pd.DataFrame, outcome: str, treatment: str, 
            selection: str) -> 'LeeBounds':
        """
        Fit Lee bounds estimator to the data.
        
        Parameters
        ----------
        data : pd.DataFrame; Input data containing outcome, treatment, and selection variables
        outcome : str; Name of the outcome variable column
        treatment : str; Name of the treatment indicator column (0=control, 1=treated)
        selection : str; Name of the selection indicator column (0=missing, 1=observed)
            
        Returns
        -------
        LeeBounds
            Self with fitted results
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
        self._validate_data(Y, D, S)
        
        # Calculate selection rates
        p1 = np.mean(S[D == 1])  # Selection rate in treated group
        p0 = np.mean(S[D == 0])  # Selection rate in control group
        
        if p1 <= p0:
            raise ValueError("Selection rate in treated group must be greater than control group")
        
        # Calculate bounds
        lower_bound, upper_bound = self._compute_bounds(Y, D, S, p1, p0)
        
        # Store results directly in self
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p1 = p1
        self.p0 = p0
        self.trim_proportion = p1 - p0
        self.n_treated = np.sum(D == 1)
        self.n_control = np.sum(D == 0)
        self.n_treated_selected = np.sum((D == 1) & (S == 1))
        self.n_control_selected = np.sum((D == 0) & (S == 1))
        
        # Compute bootstrap confidence intervals and standard errors
        bootstrap_results = self._bootstrap(data, outcome, treatment, selection)
        self.lower_bound_ci = bootstrap_results['lower_bound_ci']
        self.upper_bound_ci = bootstrap_results['upper_bound_ci']
        self.lower_bound_se = bootstrap_results['lower_bound_se']
        self.upper_bound_se = bootstrap_results['upper_bound_se']
        self.lower_bounds_bootstrap = bootstrap_results['lower_bounds']
        self.upper_bounds_bootstrap = bootstrap_results['upper_bounds']
        
        return self
    
    def summary(self) -> str:
        """
        Return a summary of the results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        if not hasattr(self, 'lower_bound'):
            return "No results available. Please fit the model first."
        
        # Format confidence intervals and standard errors
        if not np.isnan(self.lower_bound_ci[0]):
            lower_ci_str = f"[{self.lower_bound_ci[0]:.4f}, {self.lower_bound_ci[1]:.4f}]"
            upper_ci_str = f"[{self.upper_bound_ci[0]:.4f}, {self.upper_bound_ci[1]:.4f}]"
            lower_se_str = f"{self.lower_bound_se:.4f}"
            upper_se_str = f"{self.upper_bound_se:.4f}"
        else:
            lower_ci_str = "Not computed"
            upper_ci_str = "Not computed"
            lower_se_str = "Not computed"
            upper_se_str = "Not computed"
        
        summary = f"""
        Lee (2009) Treatment Effect Bounds
        =================================

        Treatment Effect Bounds:
        - Lower bound: {self.lower_bound:.4f}
        - Upper bound: {self.upper_bound:.4f}
        - Bound width: {self.upper_bound - self.lower_bound:.4f}

        Bootstrap Confidence Intervals ({int(self.ci_level*100)}%):
        - Lower bound CI: {lower_ci_str}
        - Upper bound CI: {upper_ci_str}

        Bootstrap Standard Errors:
        - Lower bound SE: {lower_se_str}
        - Upper bound SE: {upper_se_str}

        Sample Information:
        - Treated observations: {self.n_treated} (selected: {self.n_treated_selected})
        - Control observations: {self.n_control} (selected: {self.n_control_selected})
        - Selection rates: p₁ = {self.p1:.3f}, p₀ = {self.p0:.3f}
        - Trim proportion: {self.trim_proportion:.3f}
        """
        return summary