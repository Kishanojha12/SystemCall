import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_sample_data(size=100):
    """
    Load sample system call data for demonstration when real-time monitoring is not available
    
    Args:
        size (int): Number of sample data points to generate
        
    Returns:
        pandas.DataFrame: Sample system call data
    """
    # Define system call types
    call_types = ['open', 'read', 'write', 'close', 'socket', 'connect', 'accept', 'stat', 'fork', 'exec']
    categories = ['file', 'file', 'file', 'file', 'network', 'network', 'network', 'file', 'process', 'process']
    
    # Define process names
    process_names = ['python', 'bash', 'chrome', 'firefox', 'systemd', 'sshd', 'nginx', 'postgres']
    
    # Generate timestamps over the last minute
    now = datetime.now()
    timestamps = [(now - timedelta(seconds=random.uniform(0, 60))).timestamp() 
                 for _ in range(size)]
    
    # Generate random data
    data = []
    for i in range(size):
        # Select random call type and corresponding category
        call_idx = random.randint(0, len(call_types) - 1)
        call_name = call_types[call_idx]
        category = categories[call_idx]
        
        # Generate random process info
        process_name = random.choice(process_names)
        process_id = random.randint(1000, 10000)
        
        # Generate random duration (with some variation by call type)
        base_duration = {
            'open': 0.5,
            'read': 2.0,
            'write': 1.5,
            'close': 0.2,
            'socket': 1.0,
            'connect': 5.0,
            'accept': 3.0,
            'stat': 0.3,
            'fork': 10.0,
            'exec': 8.0
        }.get(call_name, 1.0)
        
        duration = max(0.01, random.gauss(base_duration, base_duration / 2))
        
        # Occasionally add an error
        error = None
        if random.random() < 0.05:  # 5% chance of error
            errors = ['EACCES', 'EINVAL', 'ENOENT', 'EBADF', 'EAGAIN']
            error = random.choice(errors)
        
        # Generate call args based on type
        args = ""
        if call_name == 'open':
            files = ['/etc/passwd', '/proc/stat', '/var/log/syslog', '/tmp/temp_file',
                     '/usr/lib/python3.8/random.py', '/home/user/documents/file.txt']
            args = random.choice(files)
        elif call_name == 'read' or call_name == 'write':
            args = f"fd={random.randint(3, 1000)}, size={random.randint(1024, 8192)}"
        elif call_name == 'socket':
            args = f"AF_INET, SOCK_STREAM, 0"
        
        # Add to dataset
        data.append({
            'timestamp': timestamps[i],
            'call_name': call_name,
            'category': category,
            'process_id': process_id,
            'process_name': process_name,
            'args': args,
            'return_value': -1 if error else random.randint(0, 1000),
            'duration_ms': duration,
            'error': error
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add datetime column
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Calculate relative time
    start_time = df['datetime'].min()
    df['relative_time'] = (df['datetime'] - start_time).dt.total_seconds()
    
    # Add performance metrics
    df['is_slow'] = df['duration_ms'] > df['duration_ms'].quantile(0.95)
    max_duration = df['duration_ms'].max()
    df['normalized_duration'] = df['duration_ms'] / max_duration if max_duration > 0 else 0
    df['has_error'] = ~df['error'].isna()
    
    # Calculate call frequencies
    process_counts = df['process_id'].value_counts()
    process_mapping = process_counts.to_dict()
    df['process_call_frequency'] = df['process_id'].map(process_mapping)
    
    call_counts = df['call_name'].value_counts()
    call_mapping = call_counts.to_dict()
    df['call_type_frequency'] = df['call_name'].map(call_mapping)
    
    return df

def plot_comparison(original_data, optimized_data, visualizer=None):
    """
    Create a comparison visualization between original and optimized system calls
    
    Args:
        original_data (pandas.DataFrame): Original system call data
        optimized_data (pandas.DataFrame): Optimized system call data
        visualizer (SystemCallVisualizer): Visualizer instance
        
    Returns:
        plotly.graph_objects.Figure: Comparison figure
    """
    if original_data is None or optimized_data is None or original_data.empty or optimized_data.empty:
        # Return empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No comparison data available",
            height=400
        )
        return fig
    
    # Create subplot with three panels
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Call Latency", "Call Frequency", "Duration Reduction"),
        column_widths=[0.35, 0.35, 0.3]
    )
    
    # Prepare data for comparison
    if 'call_name' in original_data.columns and 'call_name' in optimized_data.columns:
        # Get common call types
        orig_calls = set(original_data['call_name'].unique())
        opt_calls = set(optimized_data['call_name'].unique())
        common_calls = orig_calls & opt_calls
        
        # For each common call type, calculate metrics
        comparison_data = []
        for call in common_calls:
            orig_call_data = original_data[original_data['call_name'] == call]
            opt_call_data = optimized_data[optimized_data['call_name'] == call]
            
            orig_duration = orig_call_data['duration_ms'].mean() if not orig_call_data.empty else 0
            opt_duration = opt_call_data['duration_ms'].mean() if not opt_call_data.empty else 0
            
            orig_count = len(orig_call_data)
            opt_count = len(opt_call_data)
            
            duration_change = orig_duration - opt_duration
            duration_pct = (duration_change / orig_duration * 100) if orig_duration > 0 else 0
            
            count_change = orig_count - opt_count
            count_pct = (count_change / orig_count * 100) if orig_count > 0 else 0
            
            comparison_data.append({
                'call_name': call,
                'orig_duration': orig_duration,
                'opt_duration': opt_duration,
                'duration_change': duration_change,
                'duration_pct': duration_pct,
                'orig_count': orig_count,
                'opt_count': opt_count,
                'count_change': count_change,
                'count_pct': count_pct
            })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by original duration
        comparison_df = comparison_df.sort_values('orig_duration', ascending=False)
        
        # Limit to top 8 calls
        if len(comparison_df) > 8:
            comparison_df = comparison_df.head(8)
        
        # Only proceed if we have comparison data
        if not comparison_df.empty:
            # Add duration comparison
            fig.add_trace(
                go.Bar(
                    x=comparison_df['call_name'],
                    y=comparison_df['orig_duration'],
                    name='Original',
                    marker_color='rgba(58, 71, 80, 0.7)',
                    text=comparison_df['orig_duration'].round(2),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['call_name'],
                    y=comparison_df['opt_duration'],
                    name='Optimized',
                    marker_color='rgba(0, 128, 0, 0.7)',
                    text=comparison_df['opt_duration'].round(2),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add frequency comparison
            fig.add_trace(
                go.Bar(
                    x=comparison_df['call_name'],
                    y=comparison_df['orig_count'],
                    name='Original',
                    marker_color='rgba(58, 71, 80, 0.7)',
                    text=comparison_df['orig_count'],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['call_name'],
                    y=comparison_df['opt_count'],
                    name='Optimized',
                    marker_color='rgba(0, 128, 0, 0.7)',
                    text=comparison_df['opt_count'],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Add pie chart showing overall improvement
            total_orig_duration = (comparison_df['orig_duration'] * comparison_df['orig_count']).sum()
            total_opt_duration = (comparison_df['opt_duration'] * comparison_df['opt_count']).sum()
            
            saved_duration = total_orig_duration - total_opt_duration
            remaining_duration = total_opt_duration
            
            if total_orig_duration > 0:
                improvement_pct = saved_duration / total_orig_duration * 100
                remaining_pct = 100 - improvement_pct
                
                fig.add_trace(
                    go.Pie(
                        labels=['Saved', 'Remaining'],
                        values=[improvement_pct, remaining_pct],
                        text=[f"{improvement_pct:.1f}%", f"{remaining_pct:.1f}%"],
                        textinfo='label+text',
                        marker=dict(colors=['rgba(0, 128, 0, 0.7)', 'rgba(58, 71, 80, 0.7)']),
                        hole=0.4
                    ),
                    row=1, col=3
                )
            
            # Update layout
            fig.update_layout(
                title_text='Original vs. Optimized System Calls',
                height=400,
                barmode='group',
                legend=dict(orientation="h", y=-0.1)
            )
            
            # Update axes
            fig.update_xaxes(title_text="System Call", row=1, col=1)
            fig.update_yaxes(title_text="Avg Duration (ms)", row=1, col=1)
            
            fig.update_xaxes(title_text="System Call", row=1, col=2)
            fig.update_yaxes(title_text="Call Count", row=1, col=2)
        
        else:
            # Empty comparison data
            fig.update_layout(
                title_text='No comparable call types found',
                height=400
            )
    
    else:
        # Calculate overall metrics
        orig_total_duration = original_data['duration_ms'].sum() if 'duration_ms' in original_data.columns else 0
        opt_total_duration = optimized_data['duration_ms'].sum() if 'duration_ms' in optimized_data.columns else 0
        
        orig_call_count = len(original_data)
        opt_call_count = len(optimized_data)
        
        # Add simple bar chart for overall comparison
        fig.add_trace(
            go.Bar(
                x=['Original', 'Optimized'],
                y=[orig_total_duration, opt_total_duration],
                marker_color=['rgba(58, 71, 80, 0.7)', 'rgba(0, 128, 0, 0.7)'],
                text=[f"{orig_total_duration:.2f}", f"{opt_total_duration:.2f}"],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=['Original', 'Optimized'],
                y=[orig_call_count, opt_call_count],
                marker_color=['rgba(58, 71, 80, 0.7)', 'rgba(0, 128, 0, 0.7)'],
                text=[orig_call_count, opt_call_count],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Add pie chart showing overall improvement
        if orig_total_duration > 0:
            saved_duration = orig_total_duration - opt_total_duration
            remaining_duration = opt_total_duration
            
            improvement_pct = saved_duration / orig_total_duration * 100
            remaining_pct = 100 - improvement_pct
            
            fig.add_trace(
                go.Pie(
                    labels=['Saved', 'Remaining'],
                    values=[improvement_pct, remaining_pct],
                    text=[f"{improvement_pct:.1f}%", f"{remaining_pct:.1f}%"],
                    textinfo='label+text',
                    marker=dict(colors=['rgba(0, 128, 0, 0.7)', 'rgba(58, 71, 80, 0.7)']),
                    hole=0.4
                ),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text='Original vs. Optimized System Calls',
            height=400
        )
        
        # Update axes
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="Total Duration (ms)", row=1, col=1)
        
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Total Call Count", row=1, col=2)
    
    return fig
