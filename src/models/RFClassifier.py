"""Random Forest classifier on top of SleepVST features.

This module wraps a scikit-learn :class:`~sklearn.ensemble.RandomForestClassifier`
into a small, self-contained estimator (:class:`SleepVSTVideoRF`) for sleep-stage
classification from concatenated features ``Z + M`` (e.g. learned SleepVST
embeddings plus hand-crafted/video metadata features).

The pipeline always median-imputes missing values before the forest, and
optionally supports:
    * hyperparameter search via :class:`~sklearn.model_selection.RandomizedSearchCV`,
    * probability calibration via :class:`~sklearn.calibration.CalibratedClassifierCV`.

Configuration is centralized in the :class:`SleepVSTVideoRFConfig` dataclass.
"""

from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib


@dataclass
class SleepVSTVideoRFConfig:
    """Configuration for :class:`SleepVSTVideoRF`.

    Groups the RandomForest hyperparameters and the optional training behaviors
    (calibration and hyperparameter search) into a single dataclass.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth; ``None`` grows trees until pure.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features considered per split (``"sqrt"``,
            ``"log2"``, an int count, or a float fraction).
        n_jobs: Parallel jobs; ``-1`` uses all available cores.
        random_state: Seed for reproducibility.
        class_weight: Class weighting strategy -- ``None``, ``"balanced_subsample"``,
            or an explicit ``{class: weight}`` dict.
        calibrate: Whether to wrap the fitted model in probability calibration.
        cv_calibrate: Number of CV folds used for calibration.
        search_params: Whether to run randomized hyperparameter search.
        search_iter: Number of parameter settings sampled during search.
        search_cv: Number of CV folds used during search.
        verbose: Verbosity level passed to the search.
    """

    # RandomForest 핵심 하이퍼파라미터
    n_estimators: int = 500
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float] = "sqrt"
    n_jobs: int = -1
    random_state: int = 42

    # 학습 옵션
    class_weight: Optional[Union[str, Dict[int, float]]] = "balanced_subsample"  # None / "balanced_subsample" / dict
    calibrate: bool = False                   # 확률 보정 여부
    cv_calibrate: int = 5                     # 보정용 CV 폴드 수
    search_params: bool = False               # 랜덤 탐색 사용 여부
    search_iter: int = 25                     # 탐색 횟수
    search_cv: int = 5                        # 탐색용 CV 폴드 수
    verbose: int = 1


class SleepVSTVideoRF:
    """Random Forest sleep-stage classifier over concatenated SleepVST features.

    Wraps a median-imputation + RandomForest pipeline with convenience methods for
    fitting, prediction, evaluation, feature-importance inspection, and
    persistence. Attributes ending in ``_`` are populated after :meth:`fit`.

    Attributes:
        cfg (SleepVSTVideoRFConfig): Active configuration.
        model_ (Optional[Pipeline]): The fitted estimator (or calibrated wrapper).
        search_ (Optional[RandomizedSearchCV]): The search object, if search was run.
        n_classes_ (Optional[int]): Number of distinct classes seen during fit.
        feature_names_ (Optional[List[str]]): Feature names aligned with columns of X.
        metadata_ (Optional[Dict[str, Any]]): Arbitrary user metadata persisted with
            the model.
    """

    def __init__(self, cfg: SleepVSTVideoRFConfig = SleepVSTVideoRFConfig()):
        """Initialize the classifier.

        Args:
            cfg: Hyperparameter and training configuration. Defaults to a fresh
                :class:`SleepVSTVideoRFConfig` with library defaults.
        """
        self.cfg = cfg
        self.model_: Optional[Pipeline] = None
        self.search_: Optional[RandomizedSearchCV] = None
        self.n_classes_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.metadata_: Optional[Dict[str, Any]] = None  # metadata 저장용 속성 추가

    @staticmethod
    def _default_search_space() -> Dict[str, List[Any]]:
        """Return the default randomized-search grid over forest hyperparameters.

        Keys are namespaced with the ``clf__`` prefix to target the
        RandomForest step inside the pipeline.

        Returns:
            Dict[str, List[Any]]: Parameter name -> list of candidate values.
        """
        return {
            "clf__n_estimators": [300, 500, 800, 1200],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
        }

    def _build_base_pipeline(self) -> Pipeline:
        """Build the base pipeline: median imputation followed by a RandomForest.

        Returns:
            Pipeline: A scikit-learn pipeline with steps ``"imputer"`` and ``"clf"``,
            configured from ``self.cfg``.
        """
        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight=self.cfg.class_weight
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", rf),
        ])
        return pipe

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> "SleepVSTVideoRF":
        """Train the model on concatenated features ``Z + M``.

        Depending on ``cfg``, this may run a randomized hyperparameter search
        and/or wrap the result in probability calibration. The final estimator is
        stored in ``self.model_``.

        Args:
            X: Concatenated feature array of shape ``(n_samples, n_features)``.
            y: Target labels of shape ``(n_samples,)``.
            feature_names: Optional names for each feature column. If omitted,
                generic names (``"feature_0"``, ...) are generated.

        Returns:
            SleepVSTVideoRF: ``self``, to allow method chaining.
        """
        self.n_classes_ = len(np.unique(y))

        # feature 이름 저장 (없으면 generic 이름 생성)
        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            # 기본 feature 이름 생성
            n_features = X.shape[1]
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        base = self._build_base_pipeline()

        # 옵션: 랜덤 탐색으로 최적 하이퍼파라미터 선택, 아니면 기본 설정으로 학습
        if self.cfg.search_params:
            param_distributions = self._default_search_space()
            cv = StratifiedKFold(n_splits=self.cfg.search_cv, shuffle=True, random_state=self.cfg.random_state)
            search = RandomizedSearchCV(
                estimator=base,
                param_distributions=param_distributions,
                n_iter=self.cfg.search_iter,
                cv=cv,
                n_jobs=self.cfg.n_jobs,
                verbose=self.cfg.verbose,
                scoring="accuracy"
            )
            search.fit(X, y)
            self.search_ = search
            best = search.best_estimator_
        else:
            base.fit(X, y)
            best = base

        # 옵션: 확률 보정 래퍼로 감싸기
        if self.cfg.calibrate:
            calib = CalibratedClassifierCV(best, cv=self.cfg.cv_calibrate, n_jobs=self.cfg.n_jobs)
            calib.fit(X, y)
            self.model_ = Pipeline([("identity", SimpleImputer(strategy="median")), ("calib", calib)])
        else:
            self.model_ = best

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given features.

        Args:
            X: Concatenated feature array of shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Predicted labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given features.

        Args:
            X: Concatenated feature array of shape ``(n_samples, n_features)``.

        Returns:
            np.ndarray: Class probabilities of shape ``(n_samples, n_classes)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model_.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate predictions against ground-truth labels.

        Args:
            X: Concatenated feature array of shape ``(n_samples, n_features)``.
            y: Ground-truth labels of shape ``(n_samples,)``.

        Returns:
            Dict[str, Any]: Metrics with keys ``"accuracy"``, ``"kappa"``,
            ``"f1_macro"``, ``"f1_weighted"``, ``"f1"`` (alias of macro F1, kept
            for backward compatibility), and ``"confusion_matrix"``.
        """
        return self._metrics_from_preds(y, self.predict(X))

    @staticmethod
    def _metrics_from_preds(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, Any]:
        """Compute the standard metric dict from labels and predictions.

        Shared by :meth:`evaluate` and :meth:`get_detailed_report` so predictions
        only need to be computed once.
        """
        f1_macro = f1_score(y, y_hat, average='macro')
        return {
            "accuracy": accuracy_score(y, y_hat),
            "kappa": cohen_kappa_score(y, y_hat),
            "f1_macro": f1_macro,
            "f1_weighted": f1_score(y, y_hat, average='weighted'),
            "f1": f1_macro,  # 기존 호환성을 위해
            "confusion_matrix": confusion_matrix(y, y_hat),
        }

    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """Return Gini feature importances as a sorted DataFrame.

        Unwraps the underlying RandomForest from whichever pipeline/calibration
        structure was produced during :meth:`fit`.

        Args:
            top_k: If given, return only the top ``k`` most important features;
                otherwise return all of them.

        Returns:
            pd.DataFrame: Columns ``"feature"`` and ``"importance"``, sorted by
            descending importance.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")

        # Pipeline 구조에서 실제 RandomForest 추출 (보정 래퍼 여부에 따라 경로가 다름)
        if hasattr(self.model_, 'named_steps'):
            if 'calib' in self.model_.named_steps:
                # Calibrated classifier인 경우
                rf_model = self.model_.named_steps['calib'].base_estimator.named_steps['clf']
            else:
                # 일반 pipeline인 경우
                rf_model = self.model_.named_steps['clf']
        else:
            rf_model = self.model_

        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
            
        return importance_df

    def plot_feature_importance(self, top_k: int = 20, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
        """Plot the top feature importances as a horizontal bar chart.

        Args:
            top_k: Number of top features to display.
            figsize: Matplotlib figure size as ``(width, height)``.
            save_path: If given, save the figure to this path at 300 DPI.
        """
        importance_df = self.get_feature_importance(top_k=top_k)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(importance_df)), importance_df['importance'].values)
        plt.yticks(range(len(importance_df)), importance_df['feature'].values)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def get_detailed_report(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Build a detailed evaluation report.

        Combines the basic metrics, a per-class classification report, feature
        importances, and prediction-confidence statistics.

        Args:
            X: Concatenated feature array of shape ``(n_samples, n_features)``.
            y: Ground-truth labels of shape ``(n_samples,)``.

        Returns:
            Dict[str, Any]: Report with keys ``"basic_metrics"``,
            ``"classification_report"``, ``"feature_importance"``, and
            (when probabilities are available) ``"prediction_confidence"``.
        """
        y_hat = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(self, 'predict_proba') else None

        report = {
            'basic_metrics': self._metrics_from_preds(y, y_hat),
            'classification_report': classification_report(y, y_hat, output_dict=True),
            'feature_importance': self.get_feature_importance()
        }
        
        if y_proba is not None:
            report['prediction_confidence'] = {
                'mean_max_proba': np.mean(np.max(y_proba, axis=1)),
                'std_max_proba': np.std(np.max(y_proba, axis=1))
            }
            
        return report

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Return the stored metadata, if any.

        Returns:
            Optional[Dict[str, Any]]: The metadata dict, or ``None`` if unset.
        """
        return self.metadata_

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set the metadata to persist alongside the model.

        Args:
            metadata: Arbitrary metadata dictionary.
        """
        self.metadata_ = metadata

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist the fitted model and associated state to disk via joblib.

        Args:
            path: Destination file path.
            metadata: Optional metadata to store; overrides any previously set
                metadata when provided.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")

        # 전달된 metadata가 있으면 갱신
        if metadata is not None:
            self.metadata_ = metadata

        payload = {
            "cfg": self.cfg,
            "model": self.model_,
            "search": self.search_,
            "n_classes": self.n_classes_,
            "feature_names": self.feature_names_,
            "metadata": self.metadata_,  # metadata 추가
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "SleepVSTVideoRF":
        """Load a model previously saved with :meth:`save`.

        Args:
            path: Path to the joblib payload.

        Returns:
            SleepVSTVideoRF: A reconstructed, fitted classifier with its config,
            model, search results, class count, feature names, and metadata
            restored.
        """
        payload = joblib.load(path)
        clf = SleepVSTVideoRF(payload["cfg"])
        clf.model_ = payload["model"]
        clf.search_ = payload["search"]
        clf.n_classes_ = payload["n_classes"]
        clf.feature_names_ = payload.get("feature_names", None)
        clf.metadata_ = payload.get("metadata", None)  # metadata 로드
        return clf
