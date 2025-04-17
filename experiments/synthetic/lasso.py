import os
### Optimization code to force the use of a single thread per process ###
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
#######

from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from syclib import Matrix, SyntheticControl
from syclib.gendata import generate_sine_dataset_A, generate_sine_dataset_B
from syclib.cluster import ClusterSC

# Suppress ConvergenceWarning from sklearn
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

results_dir = Path(__file__).parent / "results" / "lasso"
os.makedirs(results_dir, exist_ok=True)

k = 2  # number of clusters
k1 = 3 # rank of subgroup A
k2 = 3 # rank of subgroup B
START_ITER = 0
END_ITER = 500

T = 10 # total time steps
T0 = 8 # pre-intervention time steps
n_As = [500, 1000]
n_Bs = [500, 1000]
noise_levels = np.arange(10, 41, 5) / 100  # in [0, 1]

noise_type = "normal"
lasso_lambda = 0.01


def process_configuration(args):
    pid, noise_level, n_A, n_B, T, T0, iteration = args

    np.random.seed(seed=iteration)

    # Generate datasets and perform analysis
    dataset_A = generate_sine_dataset_A(
        n_A, T, noise_level, num_signals=k1, noise_type=noise_type
    )  # T x n_A
    dataset_B = generate_sine_dataset_B(
        n_B, T, noise_level, num_signals=k2, noise_type=noise_type
    )  # T x n_B

    # compute and store delta
    u1, s1, vh1 = np.linalg.svd(dataset_A.T, full_matrices=False)
    u2, s2, vh2 = np.linalg.svd(dataset_B.T, full_matrices=False)
    u1_tilde = u1 @ np.diag(s1)  # n_A by k=min(n_A, T)
    u2_tilde = u2 @ np.diag(s2)
    delta = u1_tilde.mean(axis=0) - u2_tilde.mean(axis=0)
    delta_norm = np.linalg.norm(delta)

    dataset = pd.concat([dataset_A, dataset_B], axis=1)  # T by n
    dataset.columns = range(dataset.shape[1])

    target_tested = dataset_A.columns[
        : int(0.3 * n_A)
    ].tolist()  # test for the first 30% targets only

    results = {
        "target_id": target_tested,
        "bench_train": [],
        "bench_test": [],
        "cluster_train": [],
        "cluster_test": [],
        "donor_ids": [],
        "cluster_weights": [],
        "bench_weights": [],
    }

    for target_id in tqdm(
        target_tested, desc=f"noise: {noise_level}", leave=False, position=pid
    ):
        # dataset is in (T by n)
        target_data = dataset[target_id]
        target_pre = target_data[:T0]
        donor_data = dataset.drop(columns=target_id)

        # perform cluster
        csc = ClusterSC(donor_data.T)
        csc.perform_clustering(k=k)
        # get selected donor group
        cluster = csc.predict_target_cluster(target_pre)
        selected_donors = csc.get_donor_group(cluster)  # n' by T

        results["donor_ids"].append(selected_donors.index.to_list())

        cluster_dataset = pd.concat(
            [pd.DataFrame(target_data).T, selected_donors], axis=0
        )  # n'+1 by T

        # cluster + lasso
        M = Matrix(cluster_dataset.T, T0=T0, target_name=target_id)
        M.denoise(num_sv=k1, transform=False)
        cluster_SC = SyntheticControl()
        cluster_SC.fit(M.pre_donor, M.pre_target, method="lasso", lmbda=lasso_lambda)
        cluster_train_mse = cluster_SC.predict_and_mse(M.pre_donor, M.pre_target)
        cluster_test_mse = cluster_SC.predict_and_mse(M.post_donor, M.post_target)
        results["cluster_train"].append(cluster_train_mse)
        results["cluster_test"].append(cluster_test_mse)
        results["cluster_weights"].append(cluster_SC.model.coef_.tolist())

        # benchmark (use all donors + lasso)
        M = Matrix(dataset, T0, target_name=target_id)
        M.denoise(num_sv=k1 + k2, transform=False)
        bench_SC = SyntheticControl()
        bench_SC.fit(M.pre_donor, M.pre_target, method="lasso", lmbda=lasso_lambda)
        bench_train_mse = bench_SC.predict_and_mse(M.pre_donor, M.pre_target)
        bench_test_mse = bench_SC.predict_and_mse(M.post_donor, M.post_target)
        results["bench_train"].append(bench_train_mse)
        results["bench_test"].append(bench_test_mse)
        results["bench_weights"].append(bench_SC.model.coef_.tolist())

    df_results = pd.DataFrame.from_dict(results)
    df_results["noise_level"] = noise_level
    df_results["n_A"] = n_A
    df_results["n_B"] = n_B
    df_results["n"] = n_A + n_B
    df_results["T"] = T
    df_results["delta"] = delta_norm
    df_results["iteration"] = iteration

    return df_results


if __name__ == "__main__":
    for iteration in range(START_ITER, END_ITER):
        print(f"Processing iteration {iteration}/{END_ITER}")
        iteration_results = []

        for idx in range(len(n_As)):
            n_A = n_As[idx]
            n_B = n_Bs[idx]

            print(f"Processing: n_A: {n_A}, n_B: {n_B}, T: {T}, k1: {k1}, k2: {k2}")

            # Prepare arguments for parallel processing
            args_list = [
                (pid, noise_level, n_A, n_B, T, T0, iteration)
                for pid, noise_level in enumerate(noise_levels)
            ]

            # Process noise levels in parallel
            with mp.Pool(processes=len(noise_levels)) as pool:
                results = list(pool.imap(process_configuration, args_list))

            # Combine results for this configuration
            config_df = pd.concat(results, axis=0, ignore_index=True)
            iteration_results.append(config_df)

        # Combine results for this iteration
        iteration_df = pd.concat(iteration_results, axis=0, ignore_index=True)

        # Save results for this iteration
        iteration_df.to_csv(
            f"{results_dir}/iteration_{iteration}.csv",
            index=False,
        )

    print("All iterations completed.")
