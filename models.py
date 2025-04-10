import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings

class SystemCallAnalyzer:
    """AI models for analyzing system call patterns and anomalies"""
    
    def __init__(self):
        """Initialize the system call analyzer"""
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.clustering_model = None
        self.regression_model = None
        self.pca_model = None
        self.feature_columns = None
        self.last_analyzed_data = None
        
    def analyze_calls(self, data, method='Pattern Recognition'):
        """
        Analyze system calls using the specified method
        
        Args:
            data (pandas.DataFrame): Processed system call data
            method (str): Analysis method - one of:
                          'Pattern Recognition', 'Anomaly Detection', 
                          'Frequency Analysis', 'Performance Impact'
                          
        Returns:
            dict: Analysis results
        """
        if data is None or data.empty:
            return {"status": "error", "message": "No data available for analysis"}
        
        # Keep a reference to the data
        self.last_analyzed_data = data.copy()
        
        # Select the appropriate analysis method
        if method == 'Pattern Recognition':
            return self._pattern_recognition(data)
        elif method == 'Anomaly Detection':
            return self._anomaly_detection(data)
        elif method == 'Frequency Analysis':
            return self._frequency_analysis(data)
        elif method == 'Performance Impact':
            return self._performance_impact_analysis(data)
        else:
            return {"status": "error", "message": f"Unknown analysis method: {method}"}
    
    def _prepare_features(self, data):
        """
        Prepare feature matrix for ML models
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        # Select numerical features for analysis
        numerical_features = []
        
        if 'duration_ms' in data.columns:
            numerical_features.append('duration_ms')
        
        if 'process_call_frequency' in data.columns:
            numerical_features.append('process_call_frequency')
            
        if 'call_type_frequency' in data.columns:
            numerical_features.append('call_type_frequency')
            
        if 'relative_time' in data.columns:
            numerical_features.append('relative_time')
        
        # Add normalized duration if available
        if 'normalized_duration' in data.columns:
            numerical_features.append('normalized_duration')
        
        if not numerical_features:
            # If no numerical features, use dummy features
            data['dummy_feature'] = 1
            numerical_features = ['dummy_feature']
        
        # Save feature columns for later reference
        self.feature_columns = numerical_features
        
        # Extract features
        X = data[numerical_features].copy()
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def _pattern_recognition(self, data):
        """
        Identify patterns in system calls using clustering
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Pattern recognition results
        """
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Skip if insufficient data
            if X.shape[0] < 10:
                return {
                    "status": "warning",
                    "message": "Insufficient data for pattern recognition",
                    "clusters": []
                }
            
            # PCA for dimensionality reduction
            if X.shape[1] > 1:
                pca = PCA(n_components=min(X.shape[1], 2))
                X_pca = pca.fit_transform(X)
                self.pca_model = pca
            else:
                X_pca = X
            
            # Determine optimal number of clusters
            max_clusters = min(10, X.shape[0] // 5 + 1)
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, max_clusters + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X)
                    
                    if len(set(labels)) > 1:  # Check that we have at least 2 clusters
                        score = silhouette_score(X, labels)
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n_clusters
            
            # Final clustering with optimal clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            self.clustering_model = kmeans
            
            # Analyze clusters
            data_with_clusters = data.copy()
            data_with_clusters['cluster'] = labels
            
            cluster_analysis = []
            for cluster_id in range(best_n_clusters):
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                
                # Calculate cluster characteristics
                common_calls = cluster_data['call_name'].value_counts().head(3).to_dict() if 'call_name' in data.columns else {}
                avg_duration = cluster_data['duration_ms'].mean() if 'duration_ms' in data.columns else 0
                call_count = len(cluster_data)
                error_rate = (cluster_data['has_error'].sum() / call_count) * 100 if 'has_error' in data.columns else 0
                
                cluster_analysis.append({
                    'cluster_id': cluster_id,
                    'size': call_count,
                    'percentage': (call_count / len(data) * 100),
                    'common_calls': common_calls,
                    'avg_duration': avg_duration,
                    'error_rate': error_rate
                })
            
            # Look for sequence patterns within clusters
            sequence_patterns = {}
            for cluster_id in range(best_n_clusters):
                cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
                if 'call_name' in cluster_data.columns and len(cluster_data) >= 5:
                    call_sequence = cluster_data.sort_values('relative_time')['call_name'].tolist()
                    
                    # Look for repeating subsequences (simplified)
                    subsequences = {}
                    for length in range(2, min(5, len(call_sequence) - 1)):
                        for i in range(len(call_sequence) - length):
                            subseq = tuple(call_sequence[i:i+length])
                            subsequences[subseq] = subsequences.get(subseq, 0) + 1
                    
                    # Filter for subsequences that appear more than once
                    repeating = {k: v for k, v in subsequences.items() if v > 1}
                    
                    # Sort by count
                    sorted_subseq = sorted(repeating.items(), key=lambda x: x[1], reverse=True)
                    
                    if sorted_subseq:
                        sequence_patterns[cluster_id] = sorted_subseq[:3]  # Top 3 patterns
            
            # Generate interpretations
            interpretations = []
            for cluster in cluster_analysis:
                if cluster['common_calls']:
                    top_call = max(cluster['common_calls'].items(), key=lambda x: x[1])[0]
                    interpretations.append(
                        f"Cluster {cluster['cluster_id']} ({cluster['percentage']:.1f}% of calls): "
                        f"Predominantly {top_call} calls with {cluster['avg_duration']:.2f}ms avg duration."
                    )
            
            return {
                "status": "success",
                "method": "pattern_recognition",
                "clusters": cluster_analysis,
                "sequence_patterns": sequence_patterns,
                "interpretations": interpretations,
                "silhouette_score": best_score
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in pattern recognition: {str(e)}",
                "clusters": []
            }
    
    def _anomaly_detection(self, data):
        """
        Detect anomalous system calls
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Anomaly detection results
        """
        try:
            # Prepare features
            X = self._prepare_features(data)
            
            # Skip if insufficient data
            if X.shape[0] < 10:
                return {
                    "status": "warning",
                    "message": "Insufficient data for anomaly detection",
                    "anomalies": []
                }
            
            # Train isolation forest
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X)
            self.anomaly_detector = model
            
            # Predict anomalies
            anomaly_scores = model.decision_function(X)
            anomaly_predictions = model.predict(X)
            
            # Add results to data
            data_with_anomalies = data.copy()
            data_with_anomalies['anomaly_score'] = anomaly_scores
            data_with_anomalies['is_anomaly'] = anomaly_predictions == -1
            
            # Extract anomalous calls
            anomalies = data_with_anomalies[data_with_anomalies['is_anomaly']].copy()
            
            # If there are no anomalies, try with higher contamination
            if len(anomalies) == 0 and len(data) > 20:
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X)
                anomaly_scores = model.decision_function(X)
                anomaly_predictions = model.predict(X)
                
                data_with_anomalies['anomaly_score'] = anomaly_scores
                data_with_anomalies['is_anomaly'] = anomaly_predictions == -1
                
                anomalies = data_with_anomalies[data_with_anomalies['is_anomaly']].copy()
            
            # Analyze anomalies
            anomaly_results = []
            for _, row in anomalies.iterrows():
                anomaly_info = {
                    'call_name': row['call_name'] if 'call_name' in row else 'Unknown',
                    'process_name': row['process_name'] if 'process_name' in row else 'Unknown',
                    'process_id': row['process_id'] if 'process_id' in row else 0,
                    'duration_ms': row['duration_ms'] if 'duration_ms' in row else 0,
                    'anomaly_score': row['anomaly_score'],
                    'error': row['error'] if 'error' in row else None,
                    'args': row['args'] if 'args' in row else ''
                }
                
                # Determine reason for anomaly
                reasons = []
                
                if 'duration_ms' in row and 'is_slow' in row and row['is_slow']:
                    reasons.append("Unusually long duration")
                
                if 'has_error' in row and row['has_error']:
                    reasons.append("Generated an error")
                
                if 'process_call_frequency' in row and row['process_call_frequency'] < 2:
                    reasons.append("Unusual process behavior")
                
                if not reasons:
                    reasons.append("Statistical outlier")
                
                anomaly_info['reasons'] = reasons
                anomaly_results.append(anomaly_info)
            
            # Sort anomalies by score (most anomalous first)
            anomaly_results.sort(key=lambda x: x['anomaly_score'])
            
            # Summary statistics
            anomaly_count = len(anomaly_results)
            anomaly_rate = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0
            
            return {
                "status": "success",
                "method": "anomaly_detection",
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "anomalies": anomaly_results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in anomaly detection: {str(e)}",
                "anomalies": []
            }
    
    def _frequency_analysis(self, data):
        """
        Analyze system call frequency patterns
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Frequency analysis results
        """
        try:
            if 'call_name' not in data.columns:
                return {
                    "status": "error",
                    "message": "Call name data not available",
                    "frequencies": {}
                }
            
            # Get call frequencies
            call_frequencies = data['call_name'].value_counts().to_dict()
            
            # Calculate process-specific frequencies
            process_frequencies = {}
            if 'process_id' in data.columns and 'process_name' in data.columns:
                process_groups = data.groupby(['process_id', 'process_name'])
                
                for (pid, pname), group in process_groups:
                    process_key = f"{pname} (PID: {pid})"
                    process_frequencies[process_key] = group['call_name'].value_counts().to_dict()
            
            # Calculate sequential patterns
            sequence_patterns = {}
            if len(data) > 10:
                # Sort by timestamp/relative_time
                if 'relative_time' in data.columns:
                    sorted_data = data.sort_values('relative_time')
                elif 'timestamp' in data.columns:
                    sorted_data = data.sort_values('timestamp')
                else:
                    sorted_data = data
                
                # Get call sequence
                call_sequence = sorted_data['call_name'].tolist()
                
                # Analyze transitions (what call follows what)
                transitions = {}
                for i in range(len(call_sequence) - 1):
                    current_call = call_sequence[i]
                    next_call = call_sequence[i+1]
                    
                    if current_call not in transitions:
                        transitions[current_call] = {}
                    
                    transitions[current_call][next_call] = transitions[current_call].get(next_call, 0) + 1
                
                # Calculate probabilities
                transition_probs = {}
                for current_call, next_calls in transitions.items():
                    total = sum(next_calls.values())
                    transition_probs[current_call] = {
                        next_call: count / total
                        for next_call, count in next_calls.items()
                    }
                
                sequence_patterns['transitions'] = transitions
                sequence_patterns['probabilities'] = transition_probs
            
            # Time-based patterns
            time_patterns = {}
            if 'relative_time' in data.columns:
                # Divide time into bins and count calls in each bin
                max_time = data['relative_time'].max()
                num_bins = min(20, int(max_time) + 1)  # Max 20 bins
                
                if num_bins > 1:
                    bins = np.linspace(0, max_time, num_bins)
                    data['time_bin'] = pd.cut(data['relative_time'], bins)
                    time_patterns['bins'] = data.groupby('time_bin').size().to_dict()
                    
                    # Call distribution over time
                    call_time_dist = {}
                    for call in data['call_name'].unique():
                        call_data = data[data['call_name'] == call]
                        call_time_dist[call] = call_data.groupby('time_bin').size().to_dict()
                    
                    time_patterns['call_distribution'] = call_time_dist
            
            # Identify high-frequency calls
            high_freq_threshold = np.percentile(list(call_frequencies.values()), 75)
            high_freq_calls = {
                call: freq for call, freq in call_frequencies.items()
                if freq >= high_freq_threshold
            }
            
            # Identify low-frequency calls
            low_freq_threshold = np.percentile(list(call_frequencies.values()), 25)
            low_freq_calls = {
                call: freq for call, freq in call_frequencies.items()
                if freq <= low_freq_threshold
            }
            
            return {
                "status": "success",
                "method": "frequency_analysis",
                "call_frequencies": call_frequencies,
                "process_frequencies": process_frequencies,
                "sequence_patterns": sequence_patterns,
                "time_patterns": time_patterns,
                "high_frequency_calls": high_freq_calls,
                "low_frequency_calls": low_freq_calls
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in frequency analysis: {str(e)}",
                "frequencies": {}
            }
    
    def _performance_impact_analysis(self, data):
        """
        Analyze performance impact of system calls
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Performance impact analysis results
        """
        try:
            if 'duration_ms' not in data.columns:
                return {
                    "status": "error",
                    "message": "Duration data not available",
                    "performance_metrics": {}
                }
            
            # Calculate basic performance metrics by call type
            call_perf = {}
            if 'call_name' in data.columns:
                call_groups = data.groupby('call_name')
                
                for call_name, group in call_groups:
                    durations = group['duration_ms']
                    call_perf[call_name] = {
                        'count': len(group),
                        'avg_duration': durations.mean(),
                        'min_duration': durations.min(),
                        'max_duration': durations.max(),
                        'std_dev': durations.std(),
                        'p95': durations.quantile(0.95),
                        'total_time': durations.sum()
                    }
            
            # Identify slow calls (above 95th percentile duration)
            p95_duration = data['duration_ms'].quantile(0.95)
            slow_calls = data[data['duration_ms'] > p95_duration].copy()
            
            # Build regression model to predict duration based on available features
            X = None
            feature_names = []
            
            # Prepare features for regression
            if 'process_call_frequency' in data.columns:
                feature_names.append('process_call_frequency')
            
            if 'call_type_frequency' in data.columns:
                feature_names.append('call_type_frequency')
                
            if feature_names:
                X = data[feature_names]
                y = data['duration_ms']
                
                # Create and train regression model
                model = LinearRegression()
                model.fit(X, y)
                self.regression_model = model
                
                # Calculate feature importance
                feature_importance = {
                    feature: abs(coef)
                    for feature, coef in zip(feature_names, model.coef_)
                }
                
                # Make predictions and calculate error metrics
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)
                mae = np.mean(np.abs(y - y_pred))
                r2 = model.score(X, y)
                
                # Get coefficients
                coefficients = {
                    feature: coef
                    for feature, coef in zip(feature_names, model.coef_)
                }
                
                regression_results = {
                    'feature_importance': feature_importance,
                    'coefficients': coefficients,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            else:
                regression_results = {
                    'message': 'Insufficient features for regression analysis'
                }
            
            # Calculate impact metrics
            total_time = data['duration_ms'].sum()
            impact_metrics = {}
            
            if 'call_name' in data.columns:
                call_time = data.groupby('call_name')['duration_ms'].sum()
                for call, time in call_time.items():
                    impact_metrics[call] = {
                        'total_time': time,
                        'percentage': (time / total_time) * 100 if total_time > 0 else 0
                    }
            
            # Identify process impact
            process_impact = {}
            if 'process_id' in data.columns and 'process_name' in data.columns:
                proc_time = data.groupby(['process_id', 'process_name'])['duration_ms'].sum()
                for (pid, pname), time in proc_time.items():
                    process_key = f"{pname} (PID: {pid})"
                    process_impact[process_key] = {
                        'total_time': time,
                        'percentage': (time / total_time) * 100 if total_time > 0 else 0
                    }
            
            return {
                "status": "success",
                "method": "performance_impact",
                "call_performance": call_perf,
                "slow_calls": slow_calls.to_dict('records') if len(slow_calls) > 0 else [],
                "regression_analysis": regression_results,
                "impact_metrics": impact_metrics,
                "process_impact": process_impact,
                "total_time": total_time
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in performance impact analysis: {str(e)}",
                "performance_metrics": {}
            }
    
    def predict_optimization_impact(self, recommendations):
        """
        Predict the impact of optimization recommendations
        
        Args:
            recommendations (list): List of optimization recommendations
            
        Returns:
            dict: Predicted impact metrics
        """
        if not self.last_analyzed_data is not None or self.last_analyzed_data.empty:
            return {
                "status": "error",
                "message": "No data available for prediction"
            }
        
        impacts = []
        for rec in recommendations:
            # Estimate impact based on recommendation type
            if 'call_name' in rec and rec['call_name'] in self.last_analyzed_data['call_name'].values:
                call_data = self.last_analyzed_data[self.last_analyzed_data['call_name'] == rec['call_name']]
                
                # Calculate baseline metrics
                baseline_duration = call_data['duration_ms'].mean()
                baseline_count = len(call_data)
                baseline_total = call_data['duration_ms'].sum()
                
                # Predict optimized metrics based on recommendation type
                estimated_reduction = 0.0
                
                if 'optimization_type' in rec:
                    if rec['optimization_type'] == 'reduce_calls':
                        estimated_reduction = 0.4  # 40% reduction in call count
                    elif rec['optimization_type'] == 'improve_performance':
                        estimated_reduction = 0.3  # 30% reduction in duration
                    elif rec['optimization_type'] == 'batch_calls':
                        estimated_reduction = 0.5  # 50% reduction in total calls
                    elif rec['optimization_type'] == 'caching':
                        estimated_reduction = 0.7  # 70% reduction in repeated calls
                    else:
                        estimated_reduction = 0.2  # Default 20% reduction
                
                impacts.append({
                    'call_name': rec['call_name'],
                    'baseline_duration': baseline_duration,
                    'baseline_count': baseline_count,
                    'baseline_total': baseline_total,
                    'estimated_reduction': estimated_reduction,
                    'optimized_duration': baseline_duration * (1 - estimated_reduction / 2),
                    'optimized_count': baseline_count * (1 - estimated_reduction / 2),
                    'optimized_total': baseline_total * (1 - estimated_reduction),
                    'time_savings': baseline_total * estimated_reduction
                })
        
        # Calculate overall impact
        if impacts:
            total_baseline = sum(impact['baseline_total'] for impact in impacts)
            total_optimized = sum(impact['optimized_total'] for impact in impacts)
            total_savings = sum(impact['time_savings'] for impact in impacts)
            overall_reduction = (total_savings / total_baseline) * 100 if total_baseline > 0 else 0
            
            return {
                "status": "success",
                "impacts": impacts,
                "total_baseline": total_baseline,
                "total_optimized": total_optimized,
                "total_savings": total_savings,
                "overall_reduction_percentage": overall_reduction
            }
        else:
            return {
                "status": "warning",
                "message": "No specific impacts could be calculated",
                "impacts": []
            }
