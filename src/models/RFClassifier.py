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
    def __init__(self, cfg: SleepVSTVideoRFConfig = SleepVSTVideoRFConfig()):
        self.cfg = cfg
        self.model_: Optional[Pipeline] = None
        self.search_: Optional[RandomizedSearchCV] = None
        self.n_classes_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.metadata_: Optional[Dict[str, Any]] = None  # metadata 저장용 속성 추가

    @staticmethod
    def _default_search_space() -> Dict[str, List[Any]]:
        return {
            "clf__n_estimators": [300, 500, 800, 1200],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
        }

    def _build_base_pipeline(self) -> Pipeline:
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
        """
        concat된 feature (Z + M)로 모델 학습
        Args:
            X: concat된 feature array (n_samples, n_features)
            y: 타겟 레이블 (n_samples,)
            feature_names: 각 feature의 이름 리스트 (optional)
        """
        self.n_classes_ = len(np.unique(y))
        
        # feature 이름 저장
        if feature_names is not None:
            self.feature_names_ = feature_names
        else:
            # 기본 feature 이름 생성
            n_features = X.shape[1]
            self.feature_names_ = [f"feature_{i}" for i in range(n_features)]

        base = self._build_base_pipeline()

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

        if self.cfg.calibrate:
            calib = CalibratedClassifierCV(best, cv=self.cfg.cv_calibrate, n_jobs=self.cfg.n_jobs)
            calib.fit(X, y)
            self.model_ = Pipeline([("identity", SimpleImputer(strategy="median")), ("calib", calib)])
        else:
            self.model_ = best

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        concat된 feature로 예측
        Args:
            X: concat된 feature array (n_samples, n_features)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        concat된 feature로 확률 예측
        Args:
            X: concat된 feature array (n_samples, n_features)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model_.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        concat된 feature로 평가
        Args:
            X: concat된 feature array (n_samples, n_features)
            y: 타겟 레이블 (n_samples,)
        """
        y_hat = self.predict(X)
        acc = accuracy_score(y, y_hat)
        kappa = cohen_kappa_score(y, y_hat)
        f1_macro = f1_score(y, y_hat, average='macro')
        f1_weighted = f1_score(y, y_hat, average='weighted')
        cm = confusion_matrix(y, y_hat)
        
        return {
            "accuracy": acc, 
            "kappa": kappa, 
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1": f1_macro,  # 기존 호환성을 위해
            "confusion_matrix": cm
        }

    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Feature importance를 반환
        Args:
            top_k: 상위 k개 feature만 반환 (None이면 모든 feature)
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        # Pipeline에서 RandomForest 모델 추출
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
        """
        Feature importance를 시각화
        Args:
            top_k: 상위 k개 feature 표시
            figsize: 그래프 크기
            save_path: 저장 경로 (optional)
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
        """
        상세한 평가 리포트 반환
        Args:
            X: concat된 feature array (n_samples, n_features)
            y: 타겟 레이블 (n_samples,)
        """
        y_hat = self.predict(X)
        y_proba = self.predict_proba(X) if hasattr(self, 'predict_proba') else None
        
        report = {
            'basic_metrics': self.evaluate(X, y),
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
        """저장된 metadata 반환"""
        return self.metadata_

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """metadata 설정"""
        self.metadata_ = metadata

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        모델과 metadata를 함께 저장
        Args:
            path: 저장할 파일 경로
            metadata: 추가로 저장할 metadata (optional)
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        # metadata 업데이트
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
        """
        모델과 metadata를 함께 로드
        """
        payload = joblib.load(path)
        clf = SleepVSTVideoRF(payload["cfg"])
        clf.model_ = payload["model"]
        clf.search_ = payload["search"]
        clf.n_classes_ = payload["n_classes"]
        clf.feature_names_ = payload.get("feature_names", None)
        clf.metadata_ = payload.get("metadata", None)  # metadata 로드
        return clf
