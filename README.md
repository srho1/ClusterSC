# ClusterSC: Advancing Synthetic Control with Donor Selection
This repo contains both code for reproducing the experiments in the paper as well as a simple library, [syclib](syclib), which we use for our experimental code and also hope can benefit the community.

## Installation
To run the experimental code, first install the library:
```python -m pip install -e .```

## Using syclib
The library comes with three main classes, [Matrix](syclib/matrix.py), [SyntheticControl](syclib/synthetic_control.py), and [ClusterSC](syclib/cluster/cluster_sc.py). Matrix preprocesses your data into a suitable format for synthetic control (target, donor, pre/post-intervention). It also allows you to denoise your data as is done for robust synthetic control. The SyntheticControl class learns the synthetic control estimator via user-specified regression methods (currently supports OLS, Ridge, Lasso). The ClusterSC class is the bulk of our paper, and implements the algorithms described. See [tutorial.ipynb](tutorial.ipynb) for a walkthrough and the code for more details.

## Experiment Code
We provide the scripts for producing all empirical results and plots in the synthetic and housing subfolders under experiments. 

For synthetic, each python script evaluates its respective regression method. Due to the large size of the results, we only provide their aggregate values (medians for mse and pairwise improvement, as well as f1 related scores), but the code for computing them is in [plot.ipynb](experiments/synthetic/plot.ipynb). This notebook also produces all related plots.

For the housing results, we include both the raw and processed data. [data.ipynb](experiments/housing/data.ipynb) processes the raw data and produces the housing dataset plots in the appendix. The script ([exp_hpi.py](experiments/housing/exp_hpi.py)) runs the test with two choices of threshold=[0.9, 0.95] to estimate the approximate rank. Then, the script [plot_hpi.py](experiments/housing/exp_hpi.py) processes the results and makes the plot that compares SC (all donors, random subset) to ClusterSC. It can easily be modified to display results for all thresholds.
