
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SystemCallVisualizer:
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set3
        
    def create_real_time_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create real-time visualization of system calls"""
        if data.empty:
            fig = go.Figure()
            fig.update_layout(title="No data available")
            return fig
            
        call_counts = data["call_name"].value_counts()
        
        fig = px.bar(
            x=call_counts.index,
            y=call_counts.values,
            title="System Call Distribution",
            labels={"x": "Call Type", "y": "Count"}
        )
        
        return fig
        
    def create_performance_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create performance visualization"""
        if data.empty:
            fig = go.Figure()
            fig.update_layout(title="No performance data available")
            return fig
            
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Call Duration Distribution", "Calls Over Time")
        )
        
        # Duration histogram
        fig.add_trace(
            go.Histogram(x=data["duration_ms"], name="Duration"),
            row=1, col=1
        )
        
        # Calls over time
        calls_over_time = data.groupby(pd.Timestamp(data["timestamp"], unit="s").dt.floor("S")).size()
        fig.add_trace(
            go.Scatter(x=calls_over_time.index, y=calls_over_time.values, name="Call Rate"),
            row=1, col=2
        )
        
        return fig
        
    def create_anomaly_plot(self, data: pd.DataFrame, anomalies: List[Dict]) -> go.Figure:
        """Create anomaly visualization"""
        if data.empty:
            fig = go.Figure()
            fig.update_layout(title="No anomaly data available")
            return fig
            
        fig = px.scatter(
            data,
            x="timestamp",
            y="duration_ms",
            color="call_name",
            title="System Calls with Anomalies",
            labels={"timestamp": "Time", "duration_ms": "Duration (ms)"}
        )
        
        # Add anomaly markers
        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            fig.add_trace(
                go.Scatter(
                    x=anomaly_df["timestamp"],
                    y=anomaly_df["duration_ms"],
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="red"),
                    name="Anomalies"
                )
            )
            
        return fig
