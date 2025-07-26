"""
Basic example of using pyleebounds for treatment effect bounds.

This example demonstrates the basic usage of the LeeBounds estimator
with simulated data that mimics a job training program with selection bias.
"""

import sys
import os
# Add parent directory to path to import pyleebounds
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pyleebounds import LeeBounds

# Set random seed for reproducibility
np.random.seed(1988)
n=1000

#####################
# Generate simulated data
#####################

# Treatment assignment (random)
D = np.random.binomial(1, 0.5, n)

# Potential outcomes (wages)
# Control potential outcome
Y0 = np.random.normal(50, 10, n)

# Treatment effect (positive on average)
treatment_effect = np.random.normal(5, 3, n)
Y1 = Y0 + treatment_effect

# Actual outcome
Y = D * Y1 + (1 - D) * Y0

# Selection mechanism (employment)
# Higher wages increase probability of employment
# Treatment also increases employment probability
employment_prob = 0.3 + 0.4 * (Y > 50) + 0.2 * D
employment_prob = np.clip(employment_prob, 0, 1)
S = np.random.binomial(1, employment_prob, n)

# Create DataFrame
data = pd.DataFrame({
    'Y': Y,
    'D': D,
    'S': S
})

#####################
# Summary statistics #
#####################

# Display data summary
print(f"Dataset shape: {data.shape}")
print("\nData summary:")
print(data.describe())

print("\nSelection rates by treatment group:")
selection_summary = data.groupby('D')['S'].agg(['mean', 'count'])
print(selection_summary)

##################
# Fit Lee bounds #
##################

# Use fewer bootstrap replications for faster execution in example
# For production use, consider using n_bootstrap=100 or higher
lb = LeeBounds(n_bootstrap=20, ci_level=0.95)
results = lb.fit(data, outcome='Y', treatment='D', selection='S')

##################
# Display results #
##################

print(lb.summary())
