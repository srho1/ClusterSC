from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import svd, solve
from kneed import KneeLocator


from syclib import Matrix, SyntheticControl, utils


# TODO: make all the data into a T by n matrix?
class ClusterSC:
    def __init__(
        self,
        data: pd.DataFrame,  # donor data only, n by T matrix
    ):
        self.data = data
        # self.scaler: StandardScaler | None = None
        self.u_tilde: np.ndarray | None = None
        self.vh: np.ndarray | None = None
        self.cluster_model: KMeans | None = None
        self.n_clusters: int | None = None
        self.max_k = self.data.shape[0] // int(self.data.shape[0] * 0.2)

    # step 1-1. scale data
    # def _scale_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """Perform scaling on donor dataset."""

    #     # Initial scaling on donor data
    #     self.scaler = StandardScaler().fit(self.data)
    #     scaled_data = pd.DataFrame(
    #         self.scaler.transform(self.data),
    #         columns=self.data.columns,
    #         index=self.data.index,
    #     )
    #     return scaled_data

    # Step 1-2. perform SVD
    def perform_svd(self, scaled_data: pd.DataFrame) -> None:
        """Perform SVD on data and return u_tilde and vh."""
        u, s, vh = svd(scaled_data, full_matrices=False)
        u_tilde = u @ np.diag(s)
        return u_tilde, vh

    # Step 1-3. Clusterting
    def perform_clustering(self, method: str = "kmeans", **kwargs) -> None:
        # scaled_data = self._scale_data()
        scaled_data = self.data
        self.u_tilde, self.vh = self.perform_svd(scaled_data)
        self.cluster_method = method

        """Perform clustering on the left singular vectors."""
        if method == "kmeans":
            if "k" not in kwargs:
                kwargs["k"] = self.find_optimal_k("silhouette")

            self.cluster_model = KMeans(
                n_clusters=kwargs["k"],
                random_state=kwargs.get("random_state") or 0,
                n_init="auto",
            ).fit(self.u_tilde)
            self.n_clusters = self.cluster_model.n_clusters
        elif method == "spectral":
            if "k" not in kwargs:
                raise ValueError(
                    "Spectral clustering requires k, silhouette not implemented."
                )

            # build nearest neighbors model + graph manually
            # mimic what skl does under the hood with affinity='nearest_neighbors'
            self.neigh = NearestNeighbors(n_neighbors=kwargs.get("n_neighbors") or 10)
            self.neigh.fit(self.u_tilde)
            connectivity = self.neigh.kneighbors_graph(mode="connectivity")
            connectivity.setdiag(1)  # mimic include_self=True
            affinity_matrix = 0.5 * (connectivity + connectivity.T)  # symmetrize

            self.cluster_model = SpectralClustering(
                n_clusters=kwargs["k"],
                random_state=kwargs.get("random_state") or 0,
                affinity="precomputed",
            ).fit(affinity_matrix)
            self.n_clusters = self.cluster_model.n_clusters
        elif method == "knn":
            self.cluster_model = NearestNeighbors(
                n_neighbors=kwargs.get("n_neighbors") or 5
            )  # fit during target prediction
            # self.n_clusters = None
            pass
        else:
            raise ValueError("Invalid method. Choose 'kmeans'.")

    def get_donor_group(self, group: int) -> pd.DataFrame:
        # group is n by T matrix
        return self.data[self.cluster_model.labels_ == group]

    def grouping_analysis(self) -> None:
        """Perform analysis on the donor groups."""
        cluster_vh = {}
        for group in range(self.n_clusters):
            df_group = self.get_donor_group(group)
            print(f"Group {group} has {df_group.shape[0]} donors.")

            # df_group_scaled = self.scaler.transform(df_group)
            df_group_scaled = df_group
            u_tilde, vh = self.perform_svd(df_group_scaled)
            cluster_vh[group] = vh
        for g1 in range(self.n_clusters):
            for g2 in range(self.n_clusters):
                if g1 != g2:
                    gamma = np.max(
                        cosine_similarity(cluster_vh[g1][0:1], cluster_vh[g2])
                    )
                    print(f"Group {g1} vs Group {g2}")
                    print(f"maximum Cosine similarity: {gamma}")

    def singval_spectrum_analysis(self) -> None:
        """Perform analysis on the singular values."""
        S = {}
        for k in range(self.n_clusters):
            dataset = self.get_donor_group(k)
            S[k] = utils.singval_test(dataset)
        return S

    ##############################
    def find_optimal_k(self, method: str = "elbow", **kwargs) -> int:
        """Find the optimal number of clusters."""
        if method == "elbow":
            return self._elbow_method(scores, **kwargs)
        elif method == "silhouette":
            return self._silhouette_method(**kwargs)
        else:
            raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")

    def _elbow_method(self, sensitivity: float = 0.05) -> int:
        """Apply the elbow method to find optimal k."""
        scores = {}
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(
                self.u_tilde
            )
            scores[k] = -kmeans.inertia_
        x = list(scores.keys())
        y = list(scores.values())
        kneedle = KneeLocator(
            x, y, S=sensitivity, curve="convex", direction="increasing"
        )
        return kneedle.elbow if kneedle.elbow is not None else self.max_k

    def _silhouette_method(self) -> int:
        """Apply the silhouette method to find optimal k."""
        silhouette_scores = []
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(
                self.u_tilde
            )
            score = silhouette_score(self.u_tilde, kmeans.labels_)
            silhouette_scores.append(score)
        return silhouette_scores.index(max(silhouette_scores)) + 2

    #########################################

    # Step 2-1. scale the target
    def predict_target_cluster(self, target_pre: pd.Series) -> int:
        """Predict a target's cluster based on its scaled projection."""
        # target_scaled = self.scaler.transform(target.to_frame().T)
        target_scaled = np.array(target_pre).reshape(1, -1)  # (1,T0)
        T0 = len(target_pre)
        v_pre = self.vh[:, :T0]  # k by T0 (k \leq T)

        # projection1 = solve(v_pre, target_scaled.T)
        # projection1 = projection1.reshape(1, -1)

        projection2 = v_pre @ target_scaled.T
        projection2 = projection2.reshape(1, -1)
        if self.cluster_method == "kmeans":
            return self.cluster_model.predict(projection2)[0]
        elif self.cluster_method == "spectral":
            #########
            # method 1: majority vote assignment based on target projection knn
            # _, indices = self.neigh.kneighbors(projection2)
            # clusters = self.cluster_model.labels_[indices[0]]
            # cluster_counts = Counter(clusters)
            # majority_cluster, _ = cluster_counts.most_common(1)[0]
            # return majority_cluster
            #########
            # method 2: remake spectral clustering model with target projection included
            tmp_cluster_model = SpectralClustering(
                self.n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=self.neigh.n_neighbors,
                random_state=0,
            )
            all_labels = tmp_cluster_model.fit_predict(
                np.concatenate([projection2, self.u_tilde], axis=0)
            )
            target_label = all_labels[0]
            donor_labels = all_labels[1:]
            # NOTE: labels are arbitrary, so we need to align them to the original labels
            # Temporary fix: just return the donors directly for now
            return self.data[donor_labels == target_label]
        # elif self.cluster_method == "knn":
        #     _, inds = self.cluster_model.kneighbors(projection2)
        #     assert len(inds) == 1
        #     return self.data.iloc[inds[0]]
        elif self.cluster_method == "knn":
            self.cluster_model.fit(np.concatenate([projection2, self.u_tilde], axis=0))
            inds = self.cluster_model.kneighbors(return_distance=False)[0] - 1
            # assert len(inds) == 1
            # print(dists)
            try:
                return self.data.iloc[inds]
            except Exception as e:
                print(e)
                print("Data inds:", self.data.index)
                print("Cluster model inds:", inds)
                exit()
        else:
            raise ValueError(
                "self.cluster_method invalid. Choose 'kmeans' or 'spectral'. Should never get here if self.perform_clustering is run."
            )
        # return projection1, projection2

    def donor_for_target(self, target_pre: pd.Series) -> pd.DataFrame:
        """Prepare the donor dataset for a given target."""
        target_cluster = self.predict_target_cluster(target_pre)
        donor_data = self.data[self.cluster_model.labels_ == target_cluster]
        return donor_data

    # def get_M(self, target_pre: pd.Series) -> Matrix:
    #     # get T0
    #     T0 = len(target_pre)
    #     T =

    #     # prepare donor data
    #     cluster = self.predict_target_cluster(target_pre)
    #     donor_data = self.get_donor_group(cluster)

    #     # prepare the dataset
    #     cluster_dataset = pd.concat([pd.DataFrame(target_pre).T, donor_data], axis=0)
    #     M = Matrix(cluster_dataset.T, T0=T0, target_name=target_pre.name)
    #     return M

    # def get_SC(self, M: Matrix, method: str = "linreg", **kwargs) -> SyntheticControl:
    #     # fit the SC model
    #     syc = SyntheticControl()
    #     syc.fit(M.pre_donor, M.pre_target, method=method, **kwargs)
    #     return syc


# Example usage:
# df = pd.read_csv("your_data.csv")
# cluster_sc = ClusterSC(df, T0=8)
# cluster_sc.fit(k='auto', method='elbow')  # or 'silhouette'
# results = cluster_sc.evaluate()
# new_target = df.loc[some_new_id]
# sc_model = cluster_sc.predict(new_target, rank=4)
