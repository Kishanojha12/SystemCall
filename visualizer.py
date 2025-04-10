import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from datetime import datetime

class SystemCallVisualizer:
    """Visualizes system call data using various plots and charts"""
    
    def __init__(self):
        """Initialize the visualizer"""
        # Color schemes
        self.color_scheme = px.colors.qualitative.Plotly
        self.category_colors = {
            'file': '#636EFA',      # Blue
            'process': '#EF553B',   # Red
            'network': '#00CC96',   # Green
            'memory': '#AB63FA',    # Purple
            'ipc': '#FFA15A',       # Orange
            'other': '#19D3F3'      # Cyan
        }
    
    def plot_real_time_calls(self, data):
        """
        Create a real-time visualization of system calls
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            plotly.graph_objects.Figure: Real-time system call figure
        """
        if data is None or data.empty:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No system call data available",
                height=400
            )
            return fig
        
        # Group by call type
        if 'call_name' in data.columns:
            call_counts = data['call_name'].value_counts().reset_index()
            call_counts.columns = ['call_name', 'count']
            
            # Sort by count
            call_counts = call_counts.sort_values('count', ascending=False)
            
            # Limit to top 10 call types
            if len(call_counts) > 10:
                call_counts = call_counts.head(10)
            
            # Create bar chart
            fig = px.bar(
                call_counts, 
                x='call_name', 
                y='count',
                labels={'call_name': 'System Call', 'count': 'Frequency'},
                title="System Call Frequency (Top 10)",
                color='call_name',
                color_discrete_sequence=self.color_scheme
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="System Call",
                yaxis_title="Frequency",
                height=400
            )
            
        elif 'category' in data.columns:
            # If call name not available, group by category
            cat_counts = data['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            
            # Create bar chart by category
            fig = px.bar(
                cat_counts, 
                x='category', 
                y='count',
                labels={'category': 'Call Category', 'count': 'Frequency'},
                title="System Call Categories",
                color='category',
                color_discrete_map=self.category_colors
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="Call Category",
                yaxis_title="Frequency",
                height=400
            )
            
        else:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No categorizable system call data available",
                height=400
            )
        
        return fig
    
    def plot_call_distribution(self, data):
        """
        Create a visualization of system call distribution
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            plotly.graph_objects.Figure: System call distribution figure
        """
        if data is None or data.empty:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No system call data available",
                height=400
            )
            return fig
        
        # Check for duration data
        if 'duration_ms' in data.columns and 'call_name' in data.columns:
            # Group by call type and calculate statistics
            call_stats = data.groupby('call_name')['duration_ms'].agg(['mean', 'count']).reset_index()
            call_stats.columns = ['call_name', 'avg_duration', 'count']
            
            # Sort by count
            call_stats = call_stats.sort_values('count', ascending=False)
            
            # Limit to top 8 call types
            if len(call_stats) > 8:
                call_stats = call_stats.head(8)
            
            # Create subplot with two charts
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=("Call Type Distribution", "Average Duration by Call Type"),
                column_widths=[0.45, 0.55]
            )
            
            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=call_stats['call_name'],
                    values=call_stats['count'],
                    textinfo='label+percent',
                    marker=dict(colors=self.color_scheme),
                    domain=dict(x=[0, 0.45])
                ),
                row=1, col=1
            )
            
            # Add bar chart for durations
            fig.add_trace(
                go.Bar(
                    x=call_stats['call_name'],
                    y=call_stats['avg_duration'],
                    marker_color=self.color_scheme[:len(call_stats)],
                    text=call_stats['avg_duration'].round(2),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="System Call Distribution and Performance",
                height=400,
                legend=dict(orientation="h", y=-0.1)
            )
            
            # Update y-axis label for bar chart
            fig.update_yaxes(title_text="Avg Duration (ms)", row=1, col=2)
            
        elif 'category' in data.columns:
            # Create pie chart by category
            cat_counts = data['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            
            fig = px.pie(
                cat_counts, 
                names='category', 
                values='count',
                title="System Call Categories",
                color='category',
                color_discrete_map=self.category_colors
            )
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
        elif 'call_name' in data.columns:
            # Create simple pie chart by call type
            call_counts = data['call_name'].value_counts().reset_index()
            call_counts.columns = ['call_name', 'count']
            
            # Limit to top 8 call types
            if len(call_counts) > 8:
                others_count = call_counts.iloc[8:]['count'].sum()
                call_counts = call_counts.iloc[:8]
                call_counts = pd.concat([
                    call_counts,
                    pd.DataFrame([{'call_name': 'Others', 'count': others_count}])
                ], ignore_index=True)
            
            fig = px.pie(
                call_counts, 
                names='call_name', 
                values='count',
                title="System Call Distribution",
                color='call_name',
                color_discrete_sequence=self.color_scheme
            )
            
            # Update layout
            fig.update_layout(
                height=400
            )
            
        else:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No distribution data available",
                height=400
            )
        
        return fig
    
    def plot_performance_metrics(self, data):
        """
        Create a visualization of system call performance metrics
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            plotly.graph_objects.Figure: Performance metrics figure
        """
        if data is None or data.empty or 'duration_ms' not in data.columns:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No performance data available",
                height=400
            )
            return fig
        
        # Group by call type
        if 'call_name' in data.columns:
            # Calculate performance metrics
            perf_metrics = data.groupby('call_name')['duration_ms'].agg([
                'mean', 'median', 'min', 'max', 'std'
            ]).reset_index()
            
            # Sort by mean duration
            perf_metrics = perf_metrics.sort_values('mean', ascending=False)
            
            # Limit to top 8 calls
            if len(perf_metrics) > 8:
                perf_metrics = perf_metrics.head(8)
            
            # Create subplot with two charts
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "bar"}]],
                subplot_titles=("Average Duration", "Duration Variability (StdDev)"),
                column_widths=[0.5, 0.5]
            )
            
            # Add bar chart for average duration
            fig.add_trace(
                go.Bar(
                    x=perf_metrics['call_name'],
                    y=perf_metrics['mean'],
                    marker_color=self.color_scheme[:len(perf_metrics)],
                    name="Average",
                    text=perf_metrics['mean'].round(2),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Add bar chart for standard deviation
            fig.add_trace(
                go.Bar(
                    x=perf_metrics['call_name'],
                    y=perf_metrics['std'],
                    marker_color=self.color_scheme[:len(perf_metrics)],
                    name="StdDev",
                    text=perf_metrics['std'].round(2),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="System Call Performance Metrics",
                height=400,
                legend=dict(orientation="h", y=-0.1)
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Avg Duration (ms)", row=1, col=1)
            fig.update_yaxes(title_text="StdDev Duration (ms)", row=1, col=2)
            
        else:
            # Create histogram of durations
            fig = px.histogram(
                data,
                x='duration_ms',
                nbins=30,
                title="Distribution of System Call Durations",
                labels={'duration_ms': 'Duration (ms)', 'count': 'Frequency'},
                color_discrete_sequence=[self.color_scheme[0]]
            )
            
            # Add median line
            median = data['duration_ms'].median()
            fig.add_vline(x=median, line_dash='dash', line_color='red',
                         annotation_text=f"Median: {median:.2f}ms",
                         annotation_position="top right")
            
            # Update layout
            fig.update_layout(
                xaxis_title="Duration (ms)",
                yaxis_title="Frequency",
                height=400
            )
        
        return fig
    
    def plot_anomalies(self, data, anomalies):
        """
        Visualize anomalous system calls
        
        Args:
            data (pandas.DataFrame): Processed system call data
            anomalies (list): List of anomalous calls
            
        Returns:
            plotly.graph_objects.Figure: Anomaly visualization figure
        """
        if data is None or data.empty or not anomalies:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No anomaly data available",
                height=400
            )
            return fig
        
        # Create anomaly dataframe
        anomaly_df = pd.DataFrame(anomalies)
        
        if 'duration_ms' in anomaly_df.columns and 'call_name' in anomaly_df.columns:
            # Create scatter plot
            fig = px.scatter(
                anomaly_df,
                x='call_name',
                y='duration_ms',
                color='anomaly_score',
                size='duration_ms',
                hover_name='call_name',
                hover_data=['process_name', 'duration_ms', 'error', 'args'],
                title="Anomalous System Calls",
                labels={
                    'call_name': 'System Call',
                    'duration_ms': 'Duration (ms)',
                    'anomaly_score': 'Anomaly Score'
                },
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="System Call",
                yaxis_title="Duration (ms)",
                height=400
            )
            
        elif 'anomaly_score' in anomaly_df.columns:
            # Create bar chart of anomaly scores
            anomaly_df = anomaly_df.sort_values('anomaly_score')
            
            fig = px.bar(
                anomaly_df,
                x='call_name',
                y='anomaly_score',
                color='anomaly_score',
                title="Anomaly Scores by System Call",
                labels={
                    'call_name': 'System Call',
                    'anomaly_score': 'Anomaly Score'
                },
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title="System Call",
                yaxis_title="Anomaly Score",
                height=400
            )
            
        else:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Insufficient anomaly data for visualization",
                height=400
            )
        
        return fig
    
    def plot_optimization_impact(self, original, optimized):
        """
        Visualize the impact of optimization
        
        Args:
            original (dict): Original performance metrics
            optimized (dict): Optimized performance metrics
            
        Returns:
            plotly.graph_objects.Figure: Optimization impact figure
        """
        if not original or not optimized:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No optimization data available",
                height=400
            )
            return fig
        
        # Create comparison data
        comparison = []
        
        # Get common calls between original and optimized
        common_calls = set(original.keys()) & set(optimized.keys())
        
        for call in common_calls:
            if 'avg_duration' in original[call] and 'avg_duration' in optimized[call]:
                comparison.append({
                    'call_name': call,
                    'original': original[call]['avg_duration'],
                    'optimized': optimized[call]['avg_duration'],
                    'improvement': original[call]['avg_duration'] - optimized[call]['avg_duration'],
                    'percentage': ((original[call]['avg_duration'] - optimized[call]['avg_duration']) / 
                                  original[call]['avg_duration'] * 100) if original[call]['avg_duration'] > 0 else 0
                })
        
        # Convert to dataframe
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by improvement percentage
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('percentage', ascending=False)
        
            # Create bar chart
            fig = go.Figure()
            
            # Add original bars
            fig.add_trace(go.Bar(
                x=comparison_df['call_name'],
                y=comparison_df['original'],
                name='Original',
                marker_color='rgba(58, 71, 80, 0.7)',
                text=comparison_df['original'].round(2),
                textposition='auto'
            ))
            
            # Add optimized bars
            fig.add_trace(go.Bar(
                x=comparison_df['call_name'],
                y=comparison_df['optimized'],
                name='Optimized',
                marker_color='rgba(0, 128, 0, 0.7)',
                text=comparison_df['optimized'].round(2),
                textposition='auto'
            ))
            
            # Update layout
            fig.update_layout(
                title_text='System Call Optimization Impact',
                xaxis_title='System Call',
                yaxis_title='Average Duration (ms)',
                barmode='group',
                height=400
            )
            
            # Add percentage improvement annotation
            for i, row in comparison_df.iterrows():
                fig.add_annotation(
                    x=row['call_name'],
                    y=max(row['original'], row['optimized']) + 0.5,
                    text=f"{row['percentage']:.1f}%",
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=10,
                        color="green"
                    )
                )
        else:
            # Create empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No comparable optimization data available",
                height=400
            )
        
        return fig
    
    def plot_historical_trends(self, historical_data):
        """
        Visualize historical trends in system call data
        
        Args:
            historical_data (list): List of DataFrames with historical system call data
            
        Returns:
            plotly.graph_objects.Figure: Historical trends figure
        """
        if not historical_data:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No historical data available",
                height=400
            )
            return fig
        
        # Prepare data
        trend_data = []
        
        for i, data in enumerate(historical_data):
            if data is not None and not data.empty:
                # Get aggregate metrics
                if 'duration_ms' in data.columns:
                    avg_duration = data['duration_ms'].mean()
                    call_count = len(data)
                    
                    trend_data.append({
                        'session': i + 1,
                        'avg_duration': avg_duration,
                        'call_count': call_count
                    })
        
        if not trend_data:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No usable historical data available",
                height=400
            )
            return fig
        
        # Convert to dataframe
        trend_df = pd.DataFrame(trend_data)
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add duration line
        fig.add_trace(
            go.Scatter(
                x=trend_df['session'],
                y=trend_df['avg_duration'],
                name="Avg Duration",
                line=dict(color='royalblue', width=2)
            ),
            secondary_y=False,
        )
        
        # Add call count line
        fig.add_trace(
            go.Scatter(
                x=trend_df['session'],
                y=trend_df['call_count'],
                name="Call Count",
                line=dict(color='firebrick', width=2)
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title_text="System Call Trends Over Time",
            height=400,
            legend=dict(orientation="h", y=-0.1)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Monitoring Session")
        fig.update_yaxes(title_text="Avg Duration (ms)", secondary_y=False)
        fig.update_yaxes(title_text="Call Count", secondary_y=True)
        
        return fig
    
    def plot_call_sequence(self, data):
        """
        Visualize system call sequences
        
        Args:
            data (pandas.DataFrame): Processed system call data
            
        Returns:
            plotly.graph_objects.Figure: System call sequence figure
        """
        if data is None or data.empty or 'call_name' not in data.columns:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No sequence data available",
                height=400
            )
            return fig
        
        # Sort by timestamp/relative_time
        if 'relative_time' in data.columns:
            sorted_data = data.sort_values('relative_time')
        elif 'timestamp' in data.columns:
            sorted_data = data.sort_values('timestamp')
        else:
            sorted_data = data
        
        # Limit to 50 most recent calls
        if len(sorted_data) > 50:
            sorted_data = sorted_data.tail(50)
        
        # Get call sequence
        calls = sorted_data['call_name'].tolist()
        
        # Get time points
        if 'relative_time' in sorted_data.columns:
            times = sorted_data['relative_time'].tolist()
        else:
            times = list(range(len(calls)))
        
        # Get durations
        if 'duration_ms' in sorted_data.columns:
            durations = sorted_data['duration_ms'].tolist()
        else:
            durations = [1] * len(calls)
        
        # Create color map
        unique_calls = sorted_data['call_name'].unique()
        call_colors = {}
        for i, call in enumerate(unique_calls):
            call_colors[call] = self.color_scheme[i % len(self.color_scheme)]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add lines connecting points
        fig.add_trace(go.Scatter(
            x=times,
            y=[0] * len(times),
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False
        ))
        
        # Add markers for each call
        for call in unique_calls:
            call_data = sorted_data[sorted_data['call_name'] == call]
            call_times = call_data['relative_time'].tolist() if 'relative_time' in call_data.columns else list(range(len(call_data)))
            call_durations = call_data['duration_ms'].tolist() if 'duration_ms' in call_data.columns else [1] * len(call_data)
            
            fig.add_trace(go.Scatter(
                x=call_times,
                y=[0] * len(call_times),
                mode='markers',
                marker=dict(
                    size=[min(max(d * 2, 6), 20) for d in call_durations],  # Scale marker size by duration
                    color=call_colors[call],
                    line=dict(width=1, color='darkgray')
                ),
                name=call,
                hovertext=[f"{call}<br>Time: {t:.2f}s<br>Duration: {d:.2f}ms" for t, d in zip(call_times, call_durations)]
            ))
        
        # Add annotations for some calls
        shown_calls = set()
        for i, (time, call, duration) in enumerate(zip(times, calls, durations)):
            # Only show annotations for a subset of calls to avoid crowding
            if call not in shown_calls and len(shown_calls) < 10 and i % 5 == 0:
                fig.add_annotation(
                    x=time,
                    y=0.2,
                    text=call,
                    showarrow=False,
                    font=dict(
                        family="Arial",
                        size=9,
                        color=call_colors[call]
                    ),
                    textangle=90
                )
                shown_calls.add(call)
        
        # Update layout
        fig.update_layout(
            title_text="System Call Sequence",
            xaxis_title="Time (s)",
            yaxis=dict(
                showticklabels=False,
                zeroline=True,
                zerolinecolor='darkgray'
            ),
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        return fig
