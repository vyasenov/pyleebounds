
# pyleebounds

A Python package for estimating treatment effect bounds under sample selection following the method of **Lee (2009)**: *"Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects."* This approach provides nonparametric bounds on treatment effects when selection into the post-treatment sample is endogenous and differs across groups.

## Installation

You can install the package using pip:

```bash
pip install pyleebounds
````

## Features

* **Sharp treatment effect bounds** under endogenous selection
* **Handles non-random attrition or sample truncation** (e.g. post-treatment employment)
* **Automatic trimming** of treated group to match selection rates
* **Flexible interface** for specifying outcomes, treatment, and selection variables
* **Bootstrap support** for confidence intervals
* **Visualization** of upper/lower bounds and trimmed distributions

## Quick Start

```python
import pandas as pd
from pyleebounds import LeeBounds

# Load or simulate data
df = pd.read_csv("your_data.csv")

# Columns:
# 'Y' = outcome variable
# 'D' = treatment indicator (1 = treated, 0 = control)
# 'S' = selection indicator (1 = observed, 0 = missing/selected out)

# Initialize and fit Lee bounds estimator
lb = LeeBounds()
results = lb.fit(df, outcome='Y', treatment='D', selection='S')

# View summary
print(results.summary())

# Plot estimated bounds
results.plot()
```

## Examples

See the `examples/` directory for worked examples on:

* Job training and post-employment wages
* Randomized control trials with non-random attrition
* Visualization of trimmed distributions
* Bootstrap confidence intervals for bounds

## Background

### Selection Bias in Post-Treatment Outcomes

In many studies, outcomes are only observed for a selected subgroup—e.g. only employed individuals report wages, or only survivors report health status. When the probability of being observed depends on the treatment, comparing outcomes across groups can lead to **selection bias**.

For example:

* A job training program increases employment. But if we only observe wages for employed individuals, the post-treatment wage comparison may **overstate or understate** the true effect of training.

### Lee Bounds: Bounding the True Effect

Lee (2009) proposed a sharp bounding procedure that adjusts for **differential selection rates** across treatment and control groups. The key idea is:

* Assume that potential outcomes and selection are **monotonic in unobservables** (i.e., individuals more likely to be selected are also more likely to have high outcomes).
* Trim the treatment group’s outcome distribution to match the control group’s selection rate.
* Compute **bounds** on the treatment effect using the extremes of possible selection bias.

---

### Notation and Key Equations

Let:

* $Y$ be the outcome (e.g. wages)
* $D \in \{0,1\}$ indicate treatment status
* $S \in \{0,1\}$ indicate selection (1 = observed, 0 = missing)

Let:

* $p_1 = \mathbb{P}(S=1 \mid D=1)$: selection rate in the treated group
* $p_0 = \mathbb{P}(S=1 \mid D=0)$: selection rate in the control group
* Assume $p_1 > p_0$ (treated more likely to be selected)

To **equalize selection**, Lee (2009) proposes trimming the upper $p_1 - p_0$ quantile of the treated group’s outcome distribution among those with $S=1$, assuming **monotonicity**.

The **Lee bounds** on the average treatment effect on the observed subpopulation are:

$$
\underline{\Delta} = \mathbb{E}[Y \mid D=1, S=1]_{\text{lower trimmed}} - \mathbb{E}[Y \mid D=0, S=1]
$$

$$
\overline{\Delta} = \mathbb{E}[Y \mid D=1, S=1]_{\text{upper trimmed}} - \mathbb{E}[Y \mid D=0, S=1]
$$

These represent **worst-case lower and upper bounds** on the average treatment effect, given the selection bias due to $p_1 \ne p_0$.

---

### Assumptions

1. **Monotonicity in unobservables**: The unobserved factors that increase selection also increase the outcome.
2. **Exogenous treatment**: Treatment is randomly assigned or unconfounded.
3. **No support violations**: The trimmed portion of the treated distribution still overlaps with the control distribution.

---

### Confidence Intervals via Bootstrap

Since the Lee bounds involve non-differentiable operations (quantile trimming), standard variance formulas are not available. Instead, **bootstrap confidence intervals** are recommended:

1. Resample units with replacement, stratified by treatment group.
2. Compute Lee bounds for each bootstrap sample.
3. Construct percentile or bias-corrected intervals using the empirical bootstrap distribution.

The package includes built-in support for:

```python
results.bootstrap(n_bootstrap=500, ci_level=0.95)
```

---

## References

* Lee, D. S. (2009). *Training, wages, and sample selection: Estimating sharp bounds on treatment effects*. *The Review of Economic Studies*, 76(3), 1071–1102.

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