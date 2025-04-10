import streamlit as st
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from system_call_monitor import SystemCallMonitor
from data_processor import DataProcessor
from models import SystemCallAnalyzer
from visualizer import SystemCallVisualizer
from optimizer import SystemCallOptimizer
from utils import load_sample_data, plot_comparison

# Page configuration
st.set_page_config(
    page_title="System Call Optimization Using AI",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'optimized' not in st.session_state:
    st.session_state.optimized = None
if 'monitor' not in st.session_state:
    st.session_state.monitor = SystemCallMonitor()
if 'processor' not in st.session_state:
    st.session_state.processor = DataProcessor()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SystemCallAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = SystemCallVisualizer()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = SystemCallOptimizer()
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []
if 'optimization_recommendations' not in st.session_state:
    st.session_state.optimization_recommendations = []

# Helper function to toggle monitoring state
def toggle_monitoring():
    if st.session_state.monitoring:
        st.session_state.monitoring = False
    else:
        st.session_state.monitoring = True
        st.session_state.start_time = datetime.now()

# Title and introduction
st.title("System Call Optimization Using AI")
st.markdown("""
This application monitors, analyzes, and helps optimize system calls using artificial intelligence.
Monitor real-time system call activities, visualize patterns, and receive AI-powered optimization recommendations.
""")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Monitoring control
    if st.session_state.monitoring:
        if st.button("Stop Monitoring", key="stop_button", type="primary"):
            toggle_monitoring()
    else:
        if st.button("Start Monitoring", key="start_button", type="primary"):
            toggle_monitoring()
    
    st.markdown("---")
    
    # System call filtering options
    st.subheader("Monitoring Options")
    selected_calls = st.multiselect(
        "Filter System Calls",
        ["open", "read", "write", "close", "fork", "exec", "socket", "connect", "accept", "stat", "all"],
        default=["all"]
    )
    
    monitoring_duration = st.slider(
        "Monitoring Duration (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10
    )
    
    st.markdown("---")
    
    # Analysis options
    st.subheader("Analysis Options")
    analysis_method = st.selectbox(
        "AI Analysis Method",
        ["Pattern Recognition", "Anomaly Detection", "Frequency Analysis", "Performance Impact"]
    )
    
    optimization_focus = st.multiselect(
        "Optimization Focus",
        ["Latency Reduction", "Resource Usage", "Call Frequency", "Call Sequence", "All"],
        default=["All"]
    )
    
    st.markdown("---")
    
    # Run analysis button
    if st.session_state.data is not None:
        if st.button("Run AI Analysis", type="primary"):
            with st.spinner("Analyzing system calls..."):
                # Analyze the collected data
                analysis_results = st.session_state.analyzer.analyze_calls(
                    st.session_state.data, 
                    method=analysis_method
                )
                
                # Generate recommendations
                st.session_state.optimization_recommendations = st.session_state.optimizer.generate_recommendations(
                    st.session_state.data,
                    analysis_results,
                    focus=optimization_focus
                )
                
                # Generate optimized version
                st.session_state.optimized = st.session_state.optimizer.optimize_calls(
                    st.session_state.data,
                    st.session_state.optimization_recommendations
                )
                
                st.success("Analysis complete!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # System Call Monitoring Section
    st.header("System Call Monitoring")
    
    # Status indicator
    status_col1, status_col2 = st.columns([1, 4])
    with status_col1:
        if st.session_state.monitoring:
            st.markdown("### 🟢 Active")
        else:
            st.markdown("### 🔴 Inactive")
    
    with status_col2:
        if st.session_state.monitoring:
            elapsed_time = datetime.now() - st.session_state.start_time
            st.markdown(f"**Monitoring for:** {elapsed_time.seconds} seconds")
    
    # Real-time monitoring display
    monitor_container = st.container()
    
    with monitor_container:
        if st.session_state.monitoring:
            # Get current system call data
            current_data = st.session_state.monitor.get_system_calls(selected_calls)
            
            # Process the raw data
            processed_data = st.session_state.processor.process_data(current_data)
            
            # Store the processed data
            st.session_state.data = processed_data
            
            # Add to historical data
            st.session_state.historical_data.append(processed_data)
            
            # Show the data as a table
            st.subheader("Current System Calls")
            if processed_data is not None and not processed_data.empty:
                st.dataframe(processed_data.head(10), use_container_width=True)
            
            # Create real-time visualization
            fig = st.session_state.visualizer.plot_real_time_calls(processed_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Automatically stop after the specified duration
            if (datetime.now() - st.session_state.start_time).seconds >= monitoring_duration:
                st.session_state.monitoring = False
                st.info(f"Monitoring automatically stopped after {monitoring_duration} seconds")
                st.rerun()
        else:
            if st.session_state.data is not None:
                st.subheader("Last Captured System Calls")
                st.dataframe(st.session_state.data.head(10), use_container_width=True)
            else:
                st.info("No monitoring data available. Click 'Start Monitoring' to begin.")

with col2:
    # Performance Metrics
    st.header("Performance Metrics")
    
    if st.session_state.data is not None:
        # Display key metrics
        metrics_df = st.session_state.processor.calculate_metrics(st.session_state.data)
        
        # Display metrics in columns
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Total Calls", metrics_df["total_calls"])
            st.metric("Unique Call Types", metrics_df["unique_calls"])
        
        with mc2:
            st.metric("Avg. Duration (ms)", f"{metrics_df['avg_duration']:.2f}")
            st.metric("Max Duration (ms)", f"{metrics_df['max_duration']:.2f}")
        
        # Display system call distribution
        st.subheader("Call Distribution")
        dist_fig = st.session_state.visualizer.plot_call_distribution(st.session_state.data)
        st.plotly_chart(dist_fig, use_container_width=True)
    else:
        st.info("Start monitoring to see performance metrics.")

# AI Analysis and Optimization Section
st.header("AI Analysis & Optimization")

if st.session_state.optimization_recommendations:
    # Display optimization recommendations
    st.subheader("Optimization Recommendations")
    
    for i, rec in enumerate(st.session_state.optimization_recommendations):
        with st.expander(f"Recommendation {i+1}: {rec['title']}", expanded=i==0):
            st.markdown(f"**Issue**: {rec['issue']}")
            st.markdown(f"**Recommendation**: {rec['recommendation']}")
            st.markdown(f"**Estimated Impact**: {rec['impact']}")
            st.markdown(f"**Implementation Difficulty**: {rec['difficulty']}")
    
    # Display comparison if optimized data is available
    if st.session_state.optimized is not None:
        st.subheader("Performance Comparison")
        
        # Compare original vs optimized
        comp_fig = plot_comparison(
            st.session_state.data, 
            st.session_state.optimized,
            st.session_state.visualizer
        )
        st.plotly_chart(comp_fig, use_container_width=True)
        
        # Display metrics comparison in columns
        orig_metrics = st.session_state.processor.calculate_metrics(st.session_state.data)
        opt_metrics = st.session_state.processor.calculate_metrics(st.session_state.optimized)
        
        improvement = (
            (orig_metrics["avg_duration"] - opt_metrics["avg_duration"]) / 
            orig_metrics["avg_duration"] * 100
        )
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.metric("Original Avg Duration (ms)", f"{orig_metrics['avg_duration']:.2f}")
        
        with comp_col2:
            st.metric("Optimized Avg Duration (ms)", f"{opt_metrics['avg_duration']:.2f}")
        
        with comp_col3:
            st.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
else:
    if st.session_state.data is not None:
        st.info("Click 'Run AI Analysis' in the sidebar to get optimization recommendations.")
    else:
        st.info("Start monitoring and collect data to enable AI analysis.")

# Historical Data Section
if st.session_state.historical_data:
    st.header("Historical Data Analysis")
    
    # Number of historical data points to display
    if len(st.session_state.historical_data) > 1:
        # Create a time series visualization
        st.subheader("System Call Trends")
        trend_fig = st.session_state.visualizer.plot_historical_trends(st.session_state.historical_data)
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("More historical data needed to display trends. Run multiple monitoring sessions.")

# Auto-refresh when monitoring is active
if st.session_state.monitoring:
    time.sleep(1)  # Small delay
    st.rerun()
