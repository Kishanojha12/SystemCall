import pandas as pd
import numpy as np
import random
from collections import defaultdict, Counter

class SystemCallOptimizer:
    """Generates optimization recommendations and simulates optimizations"""
    
    def __init__(self):
        """Initialize the system call optimizer"""
        # Optimization techniques
        self.optimization_techniques = {
            'caching': {
                'description': 'Use caching to avoid repeated calls',
                'applicable_calls': ['open', 'read', 'stat'],
                'impact': 'high',
                'difficulty': 'medium'
            },
            'batching': {
                'description': 'Batch multiple calls into a single operation',
                'applicable_calls': ['write', 'read'],
                'impact': 'high',
                'difficulty': 'medium'
            },
            'mmap': {
                'description': 'Use memory mapping instead of read/write',
                'applicable_calls': ['read', 'write'],
                'impact': 'high',
                'difficulty': 'hard'
            },
            'buffering': {
                'description': 'Increase buffer sizes for I/O operations',
                'applicable_calls': ['read', 'write'],
                'impact': 'medium',
                'difficulty': 'easy'
            },
            'async_io': {
                'description': 'Use asynchronous I/O for non-blocking operations',
                'applicable_calls': ['read', 'write', 'connect', 'accept'],
                'impact': 'high',
                'difficulty': 'hard'
            },
            'connection_pooling': {
                'description': 'Reuse connections instead of creating new ones',
                'applicable_calls': ['socket', 'connect'],
                'impact': 'medium',
                'difficulty': 'medium'
            },
            'syscall_reduction': {
                'description': 'Reduce the number of system calls by combining operations',
                'applicable_calls': ['stat', 'open', 'close'],
                'impact': 'medium',
                'difficulty': 'medium'
            },
            'process_reuse': {
                'description': 'Reuse processes instead of frequent fork/exec',
                'applicable_calls': ['fork', 'exec'],
                'impact': 'high',
                'difficulty': 'hard'
            }
        }
    
    def generate_recommendations(self, data, analysis_results, focus=None):
        """
        Generate optimization recommendations based on analysis results
        
        Args:
            data (pandas.DataFrame): Processed system call data
            analysis_results (dict): Results from system call analysis
            focus (list): Optimization focus areas
            
        Returns:
            list: List of optimization recommendations
        """
        if data is None or data.empty:
            return []
        
        # Initialize recommendations
        recommendations = []
        
        # Check analysis method
        analysis_method = analysis_results.get('method', '')
        
        # Process based on analysis method
        if analysis_method == 'pattern_recognition':
            recommendations.extend(self._pattern_based_recommendations(data, analysis_results))
        
        elif analysis_method == 'anomaly_detection':
            recommendations.extend(self._anomaly_based_recommendations(data, analysis_results))
        
        elif analysis_method == 'frequency_analysis':
            recommendations.extend(self._frequency_based_recommendations(data, analysis_results))
        
        elif analysis_method == 'performance_impact':
            recommendations.extend(self._performance_based_recommendations(data, analysis_results))
        
        # Add general recommendations
        recommendations.extend(self._general_recommendations(data))
        
        # Filter by focus if provided
        if focus and 'All' not in focus:
            filtered_recommendations = []
            for rec in recommendations:
                # Check if recommendation matches any focus area
                matches_focus = False
                for f in focus:
                    if f.lower() in rec.get('category', '').lower():
                        matches_focus = True
                        break
                
                if matches_focus:
                    filtered_recommendations.append(rec)
            
            recommendations = filtered_recommendations
        
        # Sort recommendations by impact
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: impact_order.get(x.get('impact', 'low'), 3))
        
        return recommendations
    
    def _pattern_based_recommendations(self, data, analysis_results):
        """
        Generate recommendations based on pattern recognition
        
        Args:
            data (pandas.DataFrame): Processed system call data
            analysis_results (dict): Pattern recognition results
            
        Returns:
            list: Pattern-based recommendations
        """
        recommendations = []
        
        # Extract cluster data
        clusters = analysis_results.get('clusters', [])
        sequence_patterns = analysis_results.get('sequence_patterns', {})
        
        # Check for clustered patterns
        for cluster in clusters:
            # Check for high frequency of specific calls
            common_calls = cluster.get('common_calls', {})
            
            for call_name, count in common_calls.items():
                # Check if this call has optimization techniques
                applicable_techniques = [
                    tech for tech, details in self.optimization_techniques.items()
                    if call_name in details['applicable_calls']
                ]
                
                if applicable_techniques and count > 5:
                    # Choose the most suitable technique
                    technique = random.choice(applicable_techniques)
                    tech_details = self.optimization_techniques[technique]
                    
                    recommendations.append({
                        'title': f"Optimize frequent {call_name} calls using {technique}",
                        'call_name': call_name,
                        'issue': f"Cluster {cluster['cluster_id']} shows high frequency of {call_name} calls ({count} occurrences)",
                        'recommendation': f"Apply {technique}: {tech_details['description']}",
                        'impact': tech_details['impact'],
                        'difficulty': tech_details['difficulty'],
                        'category': 'Call Frequency',
                        'optimization_type': 'reduce_calls'
                    })
            
            # Check for high error rates
            if cluster.get('error_rate', 0) > 10:
                recommendations.append({
                    'title': f"Address high error rate in cluster {cluster['cluster_id']}",
                    'call_name': list(common_calls.keys())[0] if common_calls else "multiple",
                    'issue': f"Cluster {cluster['cluster_id']} has {cluster['error_rate']:.1f}% error rate",
                    'recommendation': "Implement proper error handling and retry logic for failed system calls",
                    'impact': 'medium',
                    'difficulty': 'medium',
                    'category': 'Error Handling',
                    'optimization_type': 'improve_reliability'
                })
            
            # Check for high durations
            if cluster.get('avg_duration', 0) > 10:  # More than 10ms average
                recommendations.append({
                    'title': f"Optimize slow calls in cluster {cluster['cluster_id']}",
                    'call_name': list(common_calls.keys())[0] if common_calls else "multiple",
                    'issue': f"Cluster {cluster['cluster_id']} has high average duration ({cluster['avg_duration']:.2f}ms)",
                    'recommendation': "Review and optimize these slow system calls, consider asynchronous alternatives",
                    'impact': 'high',
                    'difficulty': 'medium',
                    'category': 'Latency Reduction',
                    'optimization_type': 'improve_performance'
                })
        
        # Check for sequence patterns
        for cluster_id, patterns in sequence_patterns.items():
            if patterns:
                # Get top pattern
                top_pattern = patterns[0]
                
                if isinstance(top_pattern, tuple) and len(top_pattern) == 2:
                    pattern, frequency = top_pattern
                    
                    if frequency >= 3:  # Only consider patterns that repeat at least 3 times
                        call_sequence = ', '.join(pattern)
                        
                        recommendations.append({
                            'title': f"Optimize repetitive call pattern",
                            'call_name': pattern[0],
                            'issue': f"Repetitive sequence detected: {call_sequence} (repeats {frequency} times)",
                            'recommendation': "Consider combining these operations or implementing a batch processing approach",
                            'impact': 'medium',
                            'difficulty': 'medium',
                            'category': 'Call Sequence',
                            'optimization_type': 'batch_calls'
                        })
        
        return recommendations
    
    def _anomaly_based_recommendations(self, data, analysis_results):
        """
        Generate recommendations based on anomaly detection
        
        Args:
            data (pandas.DataFrame): Processed system call data
            analysis_results (dict): Anomaly detection results
            
        Returns:
            list: Anomaly-based recommendations
        """
        recommendations = []
        
        # Extract anomaly data
        anomalies = analysis_results.get('anomalies', [])
        
        # Group anomalies by call name
        anomaly_calls = defaultdict(list)
        for anomaly in anomalies:
            call_name = anomaly.get('call_name', 'Unknown')
            anomaly_calls[call_name].append(anomaly)
        
        # Generate recommendations for each type of anomalous call
        for call_name, call_anomalies in anomaly_calls.items():
            # Skip if too few anomalies
            if len(call_anomalies) < 2:
                continue
            
            # Check reasons for anomalies
            reasons = [reason for anomaly in call_anomalies for reason in anomaly.get('reasons', [])]
            reason_counts = Counter(reasons)
            
            # Check for slow duration anomalies
            if reason_counts.get("Unusually long duration", 0) >= 2:
                # Find applicable optimization techniques
                applicable_techniques = [
                    tech for tech, details in self.optimization_techniques.items()
                    if call_name in details['applicable_calls']
                ]
                
                if applicable_techniques:
                    technique = random.choice(applicable_techniques)
                    tech_details = self.optimization_techniques[technique]
                    
                    recommendations.append({
                        'title': f"Optimize slow {call_name} calls",
                        'call_name': call_name,
                        'issue': f"Found {reason_counts['Unusually long duration']} anomalously slow {call_name} calls",
                        'recommendation': f"Apply {technique}: {tech_details['description']}",
                        'impact': tech_details['impact'],
                        'difficulty': tech_details['difficulty'],
                        'category': 'Latency Reduction',
                        'optimization_type': 'improve_performance'
                    })
            
            # Check for error anomalies
            if reason_counts.get("Generated an error", 0) >= 2:
                recommendations.append({
                    'title': f"Fix errors in {call_name} calls",
                    'call_name': call_name,
                    'issue': f"Found {reason_counts['Generated an error']} {call_name} calls with errors",
                    'recommendation': "Implement proper error handling and fallback mechanisms",
                    'impact': 'medium',
                    'difficulty': 'medium',
                    'category': 'Error Handling',
                    'optimization_type': 'improve_reliability'
                })
            
            # Check for unusual process behavior
            if reason_counts.get("Unusual process behavior", 0) >= 2:
                recommendations.append({
                    'title': f"Review inconsistent {call_name} usage",
                    'call_name': call_name,
                    'issue': f"Inconsistent usage pattern for {call_name} calls across processes",
                    'recommendation': "Standardize system call usage patterns across application",
                    'impact': 'medium',
                    'difficulty': 'hard',
                    'category': 'Call Consistency',
                    'optimization_type': 'standardize_calls'
                })
        
        return recommendations
    
    def _frequency_based_recommendations(self, data, analysis_results):
        """
        Generate recommendations based on frequency analysis
        
        Args:
            data (pandas.DataFrame): Processed system call data
            analysis_results (dict): Frequency analysis results
            
        Returns:
            list: Frequency-based recommendations
        """
        recommendations = []
        
        # Extract frequency data
        call_frequencies = analysis_results.get('call_frequencies', {})
        high_frequency_calls = analysis_results.get('high_frequency_calls', {})
        sequence_patterns = analysis_results.get('sequence_patterns', {})
        
        # Check for high frequency calls
        for call_name, frequency in high_frequency_calls.items():
            if frequency >= 10:  # Only consider significantly frequent calls
                # Find applicable optimization techniques
                applicable_techniques = [
                    tech for tech, details in self.optimization_techniques.items()
                    if call_name in details['applicable_calls']
                ]
                
                if applicable_techniques:
                    technique = random.choice(applicable_techniques)
                    tech_details = self.optimization_techniques[technique]
                    
                    recommendations.append({
                        'title': f"Reduce {call_name} frequency using {technique}",
                        'call_name': call_name,
                        'issue': f"High frequency of {call_name} calls ({frequency} occurrences)",
                        'recommendation': f"Apply {technique}: {tech_details['description']}",
                        'impact': tech_details['impact'],
                        'difficulty': tech_details['difficulty'],
                        'category': 'Call Frequency',
                        'optimization_type': 'reduce_calls'
                    })
        
        # Check for common transitions/patterns
        if 'transitions' in sequence_patterns:
            transitions = sequence_patterns['transitions']
            probabilities = sequence_patterns.get('probabilities', {})
            
            for call_name, next_calls in transitions.items():
                # Find most common next call
                if next_calls:
                    next_call, count = max(next_calls.items(), key=lambda x: x[1])
                    
                    # Check if this transition is very frequent
                    if count >= 5 and call_name in self.optimization_techniques and next_call in self.optimization_techniques:
                        recommendations.append({
                            'title': f"Optimize {call_name} → {next_call} sequence",
                            'call_name': call_name,
                            'issue': f"Common call sequence: {call_name} → {next_call} (occurs {count} times)",
                            'recommendation': f"Consider combining these operations or use batch processing",
                            'impact': 'medium',
                            'difficulty': 'medium',
                            'category': 'Call Sequence',
                            'optimization_type': 'batch_calls'
                        })
        
        return recommendations
    
    def _performance_based_recommendations(self, data, analysis_results):
        """
        Generate recommendations based on performance impact analysis
        
        Args:
            data (pandas.DataFrame): Processed system call data
            analysis_results (dict): Performance impact analysis results
            
        Returns:
            list: Performance-based recommendations
        """
        recommendations = []
        
        # Extract performance data
        call_performance = analysis_results.get('call_performance', {})
        slow_calls = analysis_results.get('slow_calls', [])
        impact_metrics = analysis_results.get('impact_metrics', {})
        
        # Check for high impact calls
        high_impact_calls = []
        for call_name, metrics in impact_metrics.items():
            if metrics.get('percentage', 0) > 10:  # Calls that take >10% of total time
                high_impact_calls.append((call_name, metrics))
        
        # Sort by percentage impact
        high_impact_calls.sort(key=lambda x: x[1]['percentage'], reverse=True)
        
        # Generate recommendations for high impact calls
        for call_name, metrics in high_impact_calls[:3]:  # Focus on top 3
            # Find applicable optimization techniques
            applicable_techniques = [
                tech for tech, details in self.optimization_techniques.items()
                if call_name in details['applicable_calls']
            ]
            
            if applicable_techniques:
                technique = random.choice(applicable_techniques)
                tech_details = self.optimization_techniques[technique]
                
                recommendations.append({
                    'title': f"Optimize high-impact {call_name} calls",
                    'call_name': call_name,
                    'issue': f"{call_name} calls consume {metrics['percentage']:.1f}% of total system call time",
                    'recommendation': f"Apply {technique}: {tech_details['description']}",
                    'impact': 'high',
                    'difficulty': tech_details['difficulty'],
                    'category': 'Latency Reduction',
                    'optimization_type': 'improve_performance'
                })
        
        # Check for calls with high variability
        for call_name, perf in call_performance.items():
            if perf.get('count', 0) >= 5 and perf.get('std_dev', 0) > perf.get('avg_duration', 0):
                recommendations.append({
                    'title': f"Stabilize variable {call_name} performance",
                    'call_name': call_name,
                    'issue': f"{call_name} calls have high variability (StdDev: {perf['std_dev']:.2f}ms)",
                    'recommendation': "Investigate causes of variable performance and implement consistent patterns",
                    'impact': 'medium',
                    'difficulty': 'hard',
                    'category': 'Performance Stability',
                    'optimization_type': 'stabilize_performance'
                })
        
        return recommendations
    
    def _general_recommendations(self, data):
        """
        Generate general recommendations based on system call data
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            list: General recommendations
        """
        recommendations = []
        
        # Check for call categories if available
        if 'category' in data.columns:
            category_counts = data['category'].value_counts()
            
            # Recommend based on dominant categories
            if 'file' in category_counts and category_counts['file'] > len(data) * 0.4:
                recommendations.append({
                    'title': "Optimize file I/O operations",
                    'call_name': "multiple",
                    'issue': f"File operations dominate system calls ({category_counts['file']} calls, {category_counts['file']/len(data)*100:.1f}%)",
                    'recommendation': "Consider memory mapping files, increasing buffer sizes, or using asynchronous I/O",
                    'impact': 'high',
                    'difficulty': 'medium',
                    'category': 'Resource Usage',
                    'optimization_type': 'improve_io'
                })
            
            if 'network' in category_counts and category_counts['network'] > len(data) * 0.3:
                recommendations.append({
                    'title': "Optimize network operations",
                    'call_name': "multiple",
                    'issue': f"Network operations are frequent ({category_counts['network']} calls, {category_counts['network']/len(data)*100:.1f}%)",
                    'recommendation': "Implement connection pooling, request batching, and consider asynchronous networking",
                    'impact': 'high',
                    'difficulty': 'medium',
                    'category': 'Resource Usage',
                    'optimization_type': 'improve_networking'
                })
            
            if 'process' in category_counts and category_counts['process'] > len(data) * 0.1:
                recommendations.append({
                    'title': "Optimize process management",
                    'call_name': "multiple",
                    'issue': f"Process management calls are significant ({category_counts['process']} calls, {category_counts['process']/len(data)*100:.1f}%)",
                    'recommendation': "Consider using process pools, worker threads, or a more efficient concurrency model",
                    'impact': 'medium',
                    'difficulty': 'hard',
                    'category': 'Resource Usage',
                    'optimization_type': 'improve_concurrency'
                })
        
        # Check for error frequency
        if 'has_error' in data.columns:
            error_count = data['has_error'].sum()
            error_rate = error_count / len(data) * 100
            
            if error_rate > 5:  # More than 5% errors
                recommendations.append({
                    'title': "Improve error handling",
                    'call_name': "multiple",
                    'issue': f"High system call error rate ({error_rate:.1f}%)",
                    'recommendation': "Implement robust error handling, retries for transient errors, and fallback mechanisms",
                    'impact': 'medium',
                    'difficulty': 'medium',
                    'category': 'Error Handling',
                    'optimization_type': 'improve_reliability'
                })
        
        # Add a general recommendation about syscall reduction
        recommendations.append({
            'title': "Reduce overall system call overhead",
            'call_name': "multiple",
            'issue': "System calls have inherent overhead due to context switching between user and kernel space",
            'recommendation': "Batch operations where possible, use higher-level APIs that minimize syscalls, and consider syscall reduction techniques",
            'impact': 'medium',
            'difficulty': 'medium',
            'category': 'Call Frequency',
            'optimization_type': 'reduce_overhead'
        })
        
        return recommendations
    
    def optimize_calls(self, data, recommendations):
        """
        Simulate the impact of applying recommendations to the system calls
        
        Args:
            data (pandas.DataFrame): Processed system call data
            recommendations (list): List of optimization recommendations
            
        Returns:
            pandas.DataFrame: Simulated optimized data
        """
        if data is None or data.empty or not recommendations:
            return data
        
        # Make a copy to avoid modifying the original
        optimized = data.copy()
        
        # Apply optimizations based on recommendations
        for rec in recommendations:
            call_name = rec.get('call_name')
            optimization_type = rec.get('optimization_type')
            
            if call_name and call_name != "multiple":
                # Get calls matching this recommendation
                matching_calls = optimized[optimized['call_name'] == call_name]
                
                if len(matching_calls) == 0:
                    continue
                
                # Apply different optimizations based on type
                if optimization_type == 'reduce_calls':
                    # Simulate reducing call frequency by removing some calls
                    reduction_factor = 0.4  # Remove 40% of calls
                    indices_to_remove = matching_calls.sample(frac=reduction_factor).index
                    optimized = optimized.drop(indices_to_remove)
                
                elif optimization_type == 'improve_performance':
                    # Simulate improving call performance by reducing duration
                    improvement_factor = 0.7  # 30% reduction in duration
                    optimized.loc[optimized['call_name'] == call_name, 'duration_ms'] *= improvement_factor
                
                elif optimization_type == 'batch_calls':
                    # Simulate batching by keeping fewer calls with slightly longer duration
                    batch_factor = 0.25  # Keep 25% of calls
                    duration_increase = 1.5  # But each is 50% longer
                    
                    # Identify calls to remove and keep
                    calls_to_keep = matching_calls.sample(frac=batch_factor)
                    calls_to_remove = matching_calls.drop(calls_to_keep.index)
                    
                    # Remove the calls not kept
                    optimized = optimized.drop(calls_to_remove.index)
                    
                    # Increase duration of kept calls to represent batching
                    optimized.loc[calls_to_keep.index, 'duration_ms'] *= duration_increase
                
                elif optimization_type == 'improve_reliability':
                    # Fix error calls
                    optimized.loc[optimized['call_name'] == call_name, 'has_error'] = False
                    optimized.loc[optimized['call_name'] == call_name, 'error'] = None
                    
                    # Slight performance improvement from better error handling
                    error_improvement = 0.9  # 10% reduction in duration
                    optimized.loc[optimized['call_name'] == call_name, 'duration_ms'] *= error_improvement
            
            elif call_name == "multiple" and optimization_type:
                # Apply general optimizations
                if optimization_type == 'reduce_overhead':
                    # General overhead reduction
                    optimized['duration_ms'] *= 0.95  # 5% overall reduction
                
                elif optimization_type == 'improve_io' and 'category' in optimized.columns:
                    # IO optimization for file operations
                    optimized.loc[optimized['category'] == 'file', 'duration_ms'] *= 0.8
                
                elif optimization_type == 'improve_networking' and 'category' in optimized.columns:
                    # Network optimization
                    optimized.loc[optimized['category'] == 'network', 'duration_ms'] *= 0.75
                
                elif optimization_type == 'improve_concurrency' and 'category' in optimized.columns:
                    # Process concurrency optimization
                    optimized.loc[optimized['category'] == 'process', 'duration_ms'] *= 0.7
        
        # Recalculate derived fields
        if 'duration_ms' in optimized.columns:
            # Recalculate if call is slow (above 95th percentile)
            duration_95th = optimized['duration_ms'].quantile(0.95)
            optimized['is_slow'] = optimized['duration_ms'] > duration_95th
            
            # Recalculate normalized duration
            max_duration = optimized['duration_ms'].max() if not optimized['duration_ms'].empty else 1
            optimized['normalized_duration'] = optimized['duration_ms'] / max_duration if max_duration > 0 else 0
        
        return optimized
