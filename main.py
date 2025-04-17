
from core.data_collection import SystemCallCollector
from core.analyzer import SystemCallAnalyzer
from core.visualizer import SystemCallVisualizer
import streamlit as st
import time

def main():
    # Initialize components
    collector = SystemCallCollector()
    analyzer = SystemCallAnalyzer()
    visualizer = SystemCallVisualizer()
    
    # Streamlit UI
    st.title("System Call Analysis Dashboard")
    
    # Monitoring control
    if st.button("Start Monitoring"):
        collector.start_monitoring()
    
    # Get current data
    data = collector.get_current_data()
    
    # Analysis
    if not data.empty:
        # Show real-time visualization
        st.plotly_chart(visualizer.create_real_time_plot(data))
        
        # Perform analysis
        analysis_result = analyzer.analyze(data, method="clustering")
        
        # Show performance metrics
        if analysis_result.performance_metrics:
            st.subheader("Performance Metrics")
            st.write(analysis_result.performance_metrics)
        
        # Show anomalies if any
        if analysis_result.anomalies:
            st.subheader("Detected Anomalies")
            st.plotly_chart(visualizer.create_anomaly_plot(data, analysis_result.anomalies))

if __name__ == "__main__":
    main()
