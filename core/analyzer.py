
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class AnalysisResult:
    method: str
    clusters: Optional[List[Dict]] = None
    anomalies: Optional[List[Dict]] = None
    performance_metrics: Optional[Dict] = None
    recommendations: Optional[List[Dict]] = None

class SystemCallAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def analyze(self, data: pd.DataFrame, method: str) -> AnalysisResult:
        """Analyze system calls using specified method"""
        if method == "clustering":
            return self._perform_clustering(data)
        elif method == "anomaly":
            return self._detect_anomalies(data)
        elif method == "performance":
            return self._analyze_performance(data)
        else:
            raise ValueError(f"Unknown analysis method: {method}")

    def _perform_clustering(self, data: pd.DataFrame) -> AnalysisResult:
        """Perform clustering analysis"""
        if len(data) < 2:
            return AnalysisResult(method="clustering")
            
        features = self._prepare_features(data)
        kmeans = KMeans(n_clusters=min(5, len(data)), random_state=42)
        clusters = kmeans.fit_predict(features)
        
        cluster_info = []
        for i in range(kmeans.n_clusters_):
            mask = clusters == i
            cluster_info.append({
                "id": i,
                "size": sum(mask),
                "avg_duration": data.loc[mask, "duration_ms"].mean(),
                "common_calls": data.loc[mask, "call_name"].value_counts().to_dict()
            })
            
        return AnalysisResult(
            method="clustering",
            clusters=cluster_info
        )

    def _detect_anomalies(self, data: pd.DataFrame) -> AnalysisResult:
        """Detect anomalies in system calls"""
        if len(data) < 2:
            return AnalysisResult(method="anomaly")
            
        # Simple anomaly detection based on duration
        q3 = data["duration_ms"].quantile(0.75)
        iqr = data["duration_ms"].quantile(0.75) - data["duration_ms"].quantile(0.25)
        threshold = q3 + 1.5 * iqr
        
        anomalies = data[data["duration_ms"] > threshold].to_dict("records")
        return AnalysisResult(
            method="anomaly",
            anomalies=anomalies
        )

    def _analyze_performance(self, data: pd.DataFrame) -> AnalysisResult:
        """Analyze performance metrics"""
        if len(data) < 1:
            return AnalysisResult(method="performance")
            
        metrics = {
            "total_calls": len(data),
            "avg_duration": data["duration_ms"].mean(),
            "max_duration": data["duration_ms"].max(),
            "calls_per_second": len(data) / (data["timestamp"].max() - data["timestamp"].min())
        }
        
        return AnalysisResult(
            method="performance",
            performance_metrics=metrics
        )

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for analysis"""
        features = data[["duration_ms"]].copy()
        return self.scaler.fit_transform(features)
