"""Classical EEG baselines used by the benchmark runners.

The implementations deliberately expose the two methods named in the
experiment plan: filter-bank CSP with LDA (FBCSP-LDA) and Riemannian
tangent-space logistic regression.  They are evaluated on the exact same
protocol split as neural models, but own their established feature extractors.
"""

from __future__ import annotations

import numpy as np

from dpc_snn.preprocessing.filters import fft_bandpass


def covariance_trials(x: np.ndarray) -> np.ndarray:
    covs = []
    for trial in x:
        centered = trial - trial.mean(axis=1, keepdims=True)
        cov = centered @ centered.T
        cov /= max(1, trial.shape[1] - 1)
        cov /= np.trace(cov) + 1e-8
        covs.append(cov)
    return np.stack(covs)


class BinaryCSP:
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BinaryCSP":
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("BinaryCSP expects exactly two classes")
        covs = covariance_trials(x)
        c0 = covs[y == classes[0]].mean(axis=0)
        c1 = covs[y == classes[1]].mean(axis=0)
        # Symmetrize and regularize the class covariances for small per-subject
        # calibration sets before the generalized eigendecomposition.
        c0 = (c0 + c0.T) / 2.0
        c1 = (c1 + c1.T) / 2.0
        scale = max(float(np.trace(c0 + c1)) / max(1, c0.shape[0]), 1e-8)
        composite = c0 + c1 + np.eye(c0.shape[0], dtype=np.float32) * (1e-6 * scale)
        vals, vecs = np.linalg.eig(np.linalg.pinv(composite) @ c0)
        order = np.argsort(vals)
        n_components = min(max(1, int(self.n_components)), x.shape[1])
        n_low = n_components // 2
        n_high = n_components - n_low
        pick = np.r_[order[:n_low], order[-n_high:]]
        self.filters_ = vecs[:, pick].real.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSP is not fitted")
        projected = np.einsum("cf,nct->nft", self.filters_, x)
        var = projected.var(axis=-1)
        return np.log(var / np.maximum(var.sum(axis=1, keepdims=True), 1e-8))


class NearestCentroid:
    def __init__(self):
        self.centroids_: dict[int, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.centroids_ = {int(cls): x[y == cls].mean(axis=0) for cls in np.unique(y)}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        labels = sorted(self.centroids_)
        centers = np.stack([self.centroids_[k] for k in labels])
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        return np.asarray([labels[i] for i in dist.argmin(axis=1)], dtype=int)


class OneVsRestCSP:
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.models: dict[int, BinaryCSP] = {}
        self.clf = NearestCentroid()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OneVsRestCSP":
        features = []
        for cls in np.unique(y):
            binary = (y == cls).astype(int)
            csp = BinaryCSP(self.n_components).fit(x, binary)
            self.models[int(cls)] = csp
            features.append(csp.transform(x))
        feat = np.concatenate(features, axis=1)
        self.clf.fit(feat, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        feat = np.concatenate([model.transform(x) for _, model in sorted(self.models.items())], axis=1)
        return self.clf.predict(feat)


class FBCSPLDA:
    """Filter-bank one-vs-rest CSP followed by shrinkage LDA.

    This is a real FBCSP-style reference baseline rather than the former
    nearest-centroid placeholder.  Bands, component count, and sampling rate
    are recorded by the caller in the run artefacts.
    """

    def __init__(
        self,
        sfreq: float,
        bands: dict[str, list[float] | tuple[float, float]],
        n_components: int = 4,
    ) -> None:
        self.sfreq = float(sfreq)
        self.bands = {str(name): (float(lo), float(hi)) for name, (lo, hi) in bands.items()}
        self.n_components = int(n_components)
        self.band_models_: dict[str, list[BinaryCSP]] = {}
        self.classes_: np.ndarray | None = None
        self.clf: object | None = None

    def _features(self, x: np.ndarray, fit: bool) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("FBCSPLDA is not fitted")
        blocks = []
        for name, (low, high) in self.bands.items():
            filtered = fft_bandpass(x, self.sfreq, low, high)
            if fit:
                models = []
                for cls in self.classes_:
                    binary = (self._fit_y == cls).astype(np.int64)
                    model = BinaryCSP(self.n_components).fit(filtered, binary)
                    models.append(model)
                self.band_models_[name] = models
            models = self.band_models_.get(name)
            if not models:
                raise RuntimeError(f"FBCSP band {name!r} is not fitted")
            blocks.extend(model.transform(filtered) for model in models)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("FBCSP-LDA requires at least two classes")
        if x.ndim != 3 or x.shape[0] != y.size:
            raise ValueError(f"Expected X=[trials, channels, time] and matching y, got {x.shape}, {y.shape}")
        self._fit_y = y
        features = self._features(x, fit=True)
        del self._fit_y
        self.clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self.clf.fit(features, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("FBCSPLDA is not fitted")
        return np.asarray(self.clf.predict(self._features(np.asarray(x, dtype=np.float32), fit=False)), dtype=np.int64)


class RiemannianLogisticRegression:
    """Covariance tangent-space logistic-regression baseline via pyRiemann."""

    def __init__(self, covariance_estimator: str = "oas", c: float = 1.0, max_iter: int = 1000) -> None:
        self.covariance_estimator = covariance_estimator
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.covariances_: object | None = None
        self.tangent_space_: object | None = None
        self.classifier_: object | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RiemannianLogisticRegression":
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if np.unique(y).size < 2:
            raise ValueError("Riemannian tangent-space logistic regression requires at least two classes")
        self.covariances_ = Covariances(estimator=self.covariance_estimator)
        self.tangent_space_ = TangentSpace(metric="riemann")
        features = self.tangent_space_.fit_transform(self.covariances_.fit_transform(x))
        self.classifier_ = LogisticRegression(C=self.c, max_iter=self.max_iter, solver="lbfgs")
        self.classifier_.fit(features, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.covariances_ is None or self.tangent_space_ is None or self.classifier_ is None:
            raise RuntimeError("RiemannianLogisticRegression is not fitted")
        cov = self.covariances_.transform(np.asarray(x, dtype=np.float32))
        return np.asarray(self.classifier_.predict(self.tangent_space_.transform(cov)), dtype=np.int64)
