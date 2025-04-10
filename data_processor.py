import pandas as pd
import numpy as np
from datetime import datetime
import statistics

class DataProcessor:
    """Process and transform raw system call data"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def process_data(self, data):
        """
        Process raw system call data
        
        Args:
            data (pandas.DataFrame): Raw system call data
            
        Returns:
            pandas.DataFrame: Processed data
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        processed = data.copy()
        
        # Convert timestamp to datetime
        if 'timestamp' in processed.columns:
            processed['datetime'] = pd.to_datetime(processed['timestamp'], unit='s')
        
        # Calculate relative time (seconds since start)
        if 'datetime' in processed.columns:
            start_time = processed['datetime'].min()
            processed['relative_time'] = (processed['datetime'] - start_time).dt.total_seconds()
        
        # Calculate efficiency metrics
        if 'duration_ms' in processed.columns:
            # Flag potential slow calls (above 95th percentile)
            duration_95th = processed['duration_ms'].quantile(0.95)
            processed['is_slow'] = processed['duration_ms'] > duration_95th
            
            # Normalize duration for comparative analysis (0-1 scale)
            max_duration = processed['duration_ms'].max() if not processed['duration_ms'].empty else 1
            processed['normalized_duration'] = processed['duration_ms'] / max_duration if max_duration > 0 else 0
        
        # Add flags for errors
        if 'error' in processed.columns:
            processed['has_error'] = ~processed['error'].isna()
        
        # Calculate call frequency by process
        if 'process_id' in processed.columns:
            process_counts = processed['process_id'].value_counts()
            process_mapping = process_counts.to_dict()
            processed['process_call_frequency'] = processed['process_id'].map(process_mapping)
        
        # Calculate call type frequency
        if 'call_name' in processed.columns:
            call_counts = processed['call_name'].value_counts()
            call_mapping = call_counts.to_dict()
            processed['call_type_frequency'] = processed['call_name'].map(call_mapping)
        
        return processed
    
    def calculate_metrics(self, data):
        """
        Calculate performance metrics from processed system call data
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if data is None or data.empty:
            return {
                "total_calls": 0,
                "unique_calls": 0,
                "avg_duration": 0,
                "max_duration": 0,
                "error_rate": 0,
                "call_rate": 0
            }
        
        # Basic metrics
        total_calls = len(data)
        unique_calls = data['call_name'].nunique() if 'call_name' in data.columns else 0
        
        # Duration metrics
        avg_duration = data['duration_ms'].mean() if 'duration_ms' in data.columns else 0
        max_duration = data['duration_ms'].max() if 'duration_ms' in data.columns else 0
        p95_duration = data['duration_ms'].quantile(0.95) if 'duration_ms' in data.columns else 0
        
        # Error metrics
        error_rate = (data['has_error'].sum() / total_calls) * 100 if 'has_error' in data.columns else 0
        
        # Call rate metrics
        if 'relative_time' in data.columns and not data['relative_time'].empty:
            time_span = data['relative_time'].max() - data['relative_time'].min()
            call_rate = total_calls / time_span if time_span > 0 else 0
        else:
            call_rate = 0
        
        # Process metrics
        top_processes = {}
        if 'process_name' in data.columns and 'process_id' in data.columns:
            proc_counts = data.groupby(['process_id', 'process_name']).size().reset_index(name='count')
            proc_counts = proc_counts.sort_values('count', ascending=False).head(5)
            for _, row in proc_counts.iterrows():
                top_processes[f"{row['process_name']} (PID: {row['process_id']})"] = row['count']
        
        # Call type metrics
        call_distribution = {}
        if 'call_name' in data.columns:
            call_counts = data['call_name'].value_counts()
            for call_name, count in call_counts.items():
                call_distribution[call_name] = count
        
        # Category metrics
        category_distribution = {}
        if 'category' in data.columns:
            cat_counts = data['category'].value_counts()
            for cat_name, count in cat_counts.items():
                category_distribution[cat_name] = count
        
        # Slowest calls
        slowest_calls = []
        if 'duration_ms' in data.columns and 'call_name' in data.columns:
            slow_data = data.sort_values('duration_ms', ascending=False).head(5)
            for _, row in slow_data.iterrows():
                slowest_calls.append({
                    'call_name': row['call_name'],
                    'duration_ms': row['duration_ms'],
                    'process_name': row.get('process_name', 'Unknown'),
                    'args': row.get('args', '')
                })
        
        return {
            "total_calls": total_calls,
            "unique_calls": unique_calls,
            "avg_duration": avg_duration,
            "max_duration": max_duration,
            "p95_duration": p95_duration,
            "error_rate": error_rate,
            "call_rate": call_rate,
            "top_processes": top_processes,
            "call_distribution": call_distribution,
            "category_distribution": category_distribution,
            "slowest_calls": slowest_calls
        }
    
    def identify_patterns(self, data):
        """
        Identify patterns in system call data
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            dict: Dictionary of identified patterns
        """
        if data is None or data.empty:
            return {}
        
        patterns = {}
        
        # Look for call sequences (common pairs of calls that occur together)
        if 'call_name' in data.columns and len(data) > 1:
            # Shift data to look at pairs of calls
            data['next_call'] = data['call_name'].shift(-1)
            call_pairs = data.groupby(['call_name', 'next_call']).size().reset_index(name='count')
            call_pairs = call_pairs.sort_values('count', ascending=False).head(10)
            
            patterns['common_call_pairs'] = []
            for _, row in call_pairs.iterrows():
                if pd.notna(row['next_call']):  # Filter out NaN values
                    patterns['common_call_pairs'].append({
                        'first_call': row['call_name'],
                        'second_call': row['next_call'],
                        'frequency': row['count']
                    })
        
        # Look for repetitive patterns in process behavior
        if 'process_id' in data.columns and 'call_name' in data.columns:
            process_patterns = {}
            for pid, group in data.groupby('process_id'):
                if len(group) > 5:  # Only analyze processes with enough calls
                    call_sequence = group['call_name'].tolist()
                    # Find repeated subsequences (simplified approach)
                    process_patterns[pid] = self._find_repeated_subsequences(call_sequence)
            
            patterns['process_patterns'] = process_patterns
        
        # Identify potential inefficiencies
        if 'duration_ms' in data.columns and 'call_name' in data.columns:
            call_duration_stats = data.groupby('call_name')['duration_ms'].agg(['mean', 'std', 'count']).reset_index()
            
            # Filter for calls with high variability and sufficient count
            variable_calls = call_duration_stats[(call_duration_stats['std'] > call_duration_stats['mean']) & 
                                                (call_duration_stats['count'] >= 5)]
            
            patterns['variable_duration_calls'] = []
            for _, row in variable_calls.iterrows():
                patterns['variable_duration_calls'].append({
                    'call_name': row['call_name'],
                    'mean_duration': row['mean'],
                    'std_dev': row['std'],
                    'count': row['count']
                })
        
        return patterns
    
    def _find_repeated_subsequences(self, sequence, min_length=2, min_repetitions=2):
        """
        Find repeated subsequences in a list
        
        Args:
            sequence (list): List to analyze
            min_length (int): Minimum subsequence length to consider
            min_repetitions (int): Minimum number of repetitions to consider
            
        Returns:
            list: List of repeated subsequences
        """
        if not sequence or len(sequence) < min_length * min_repetitions:
            return []
        
        results = []
        
        # Look for subsequences of different lengths
        for length in range(min_length, min(len(sequence) // min_repetitions + 1, 6)):  # Limit to reasonable lengths
            # Count occurrences of each subsequence
            subsequence_counts = {}
            
            for i in range(len(sequence) - length + 1):
                # Convert subsequence to tuple so it can be used as dict key
                subseq = tuple(sequence[i:i+length])
                subsequence_counts[subseq] = subsequence_counts.get(subseq, 0) + 1
            
            # Filter for subsequences that repeat enough times
            repeated = [(list(subseq), count) for subseq, count in subsequence_counts.items() 
                       if count >= min_repetitions]
            
            # Sort by count (descending)
            repeated.sort(key=lambda x: x[1], reverse=True)
            
            # Add top repeated subsequences to results
            results.extend(repeated[:3])
        
        # Sort results by repetition count and return
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:5]  # Return top 5 results
