# pyleebounds

![](https://img.shields.io/badge/license-MIT-green)

A Python package for estimating treatment effect bounds under sample selection, based on the method of Lee (2009). This approach is especially useful when selection into the observed sample (e.g., post-treatment employment) differs by treatment status and may introduce bias in outcome comparisons.

## Installation

You can install the package using pip:

```bash
pip install pyleebounds
````

## Features

* Sharp nonparametric bounds on treatment effects under endogenous sample selection
* Automatically handles non-random attrition or truncation (e.g. only observing outcomes for employed individuals)
* Bootstrap confidence intervals
* Seamless integration with `pandas`

## Quick Start

```python
import pandas as pd
import numpy as np
from pyleebounds import LeeBounds

# Generate synthetic data
np.random.seed(1988)
n = 1000

# Treatment assignment (random)
D = np.random.binomial(1, 0.5, n)

# Potential outcomes (e.g., wages)
Y0 = np.random.normal(50, 10, n)  # Control potential outcome
treatment_effect = np.random.normal(5, 3, n)  # Treatment effect
Y1 = Y0 + treatment_effect  # Treated potential outcome
Y = D * Y1 + (1 - D) * Y0  # Actual outcome

# Selection mechanism (e.g., employment)
# Higher wages and treatment increase employment probability
employment_prob = 0.3 + 0.4 * (Y > 50) + 0.2 * D
employment_prob = np.clip(employment_prob, 0, 1)
S = np.random.binomial(1, employment_prob, n)

# Create DataFrame
df = pd.DataFrame({
    'Y': Y,  # outcome variable
    'D': D,  # treatment indicator (1 = treated, 0 = control)
    'S': S   # selection indicator (1 = observed, 0 = missing/selected out)
})

# Initialize and fit Lee bounds estimator
# Use fewer bootstrap replications for faster execution in this example
lb = LeeBounds(n_bootstrap=20, ci_level=0.95)
results = lb.fit(df, outcome='Y', treatment='D', selection='S')

# View comprehensive summary
print(lb.summary())
```


## Examples

You can find detailed usage examples in the `examples/` directory.

## Background

### Why Treatment Bounds?

In many applied settings, outcomes are observed only for a selected subset of the population—e.g., wages are observed only for employed individuals. If treatment affects selection (e.g., job training increases employment), naïvely comparing outcomes may confound treatment effects with selection effects.

Lee (2009) offers a way to partially identify treatment effects by trimming the treated group's distribution to match the control group’s selection rate under plausible assumptions.

---

### Notation

Let's establish the following notation:

* $Y$: observed *continuous* outcome
* $D \in \{0,1\}$: treatment indicator (1 = treated)
* $S \in \{0,1\}$: selection indicator (1 = observed)
* $Y(0), Y(1)$: potential outcomes under control/treatment
* $S(0), S(1)$: potential selection statuses
* $p_1 = \Pr(S=1 \mid D=1)$, $p_0 = \Pr(S=1 \mid D=0)$: selection rates

For each unit we observe $\left(D, S, Y\times S \right)$. 

---

### Assumptions

1. Monotonicity: Treatment weakly increases the probability of being observed $$S(1)\geq S(0).$$
2. Exogeneity: Treatment is randomly assigned or unconfounded $$\left(Y(0),Y(1),S(0),S(1)\right) \perp D.$$

---

### Main Result

To adjust for differential selection, Lee (2009) suggested trimming the treated group’s outcome distribution among those with $S=1$. We then compute bounds on the average treatment effect (ATE) for the observed sample as:

$$
ATE \in \left[ \underline{\Delta}, \overline{\Delta} \right],
$$

where:

$$
\underline{\Delta} = \mathbb{E}[Y \mid Y\geq q^{1-\frac{p_0}{p_1}}, D=1, S=1] - \mathbb{E}[Y \mid D=0, S=1]
$$

$$
\overline{\Delta} = \mathbb{E}[Y \mid Y\leq q^{\frac{p_0}{p_1}}, D=1, S=1] - \mathbb{E}[Y \mid D=0, S=1]
$$

Here $q^{u}$ represents the $u$th quantile of $Y|D=1,S=1$. These form sharp bounds under the stated assumptions.

These bounds can be tightened in presence of additional covariates $X$, but this package does not offer that functionality. See also Semenova (2020).

---

### Confidence Intervals

Since the Lee bounds involve non-differentiable operations (quantile trimming), variance formulas are complex. Instead, this package provides bootstrap confidence intervals computed as follows:

1. Resample units with replacement, stratified by treatment group.
2. Compute Lee bounds for each bootstrap sample.
3. Construct percentile intervals using the empirical bootstrap distribution.

## References

* Lee, D. S. (2009). *Training, wages, and sample selection: Estimating sharp bounds on treatment effects*. *The Review of Economic Studies*, 76(3), 1071–1102.
* Semenova, V. (2020). Generalized lee bounds. arXiv preprint arXiv:2008.12720.
* Tauchmann, H. (2014). Lee (2009) treatment-effect bounds for nonrandom sample selection. The Stata Journal, 14(4), 884-894.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Citation

To cite this package in publications, use the following BibTeX entry:

```bibtex
@misc{yasenov2025pyleebounds,
  author       = {Vasco Yasenov},
  title        = {pyleebounds: Python Tools for Estimating Treatment Effect Bounds under Sample Selection},
  year         = {2025},
  howpublished = {\url{https://github.com/vyasenov/pyleebounds}},
  note         = {Version 0.1.0}
}
```