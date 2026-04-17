"""
Streamlit Monitoring Dashboard
===============================

Real-time monitoring interface for the Smart Flood Management System.

Features:
- Live water level monitoring with maps
- Prediction visualization with confidence bands
- Physics residual monitoring
- RL decision display
- Alert management
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta


def launch_monitoring_dashboard(system=None, port: int = 8501, config: dict = None):
    """
    Launch Streamlit dashboard.
    
    This is the entry point - the actual dashboard code runs
    when streamlit executes this file.
    """
    st.set_page_config(
        page_title="Smart Yellow River Flood Management",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title
    st.title("🌊 Smart Yellow River Flood Management System")
    st.markdown("""
    **Real-time intelligent flood prevention decision support system**
    
    Integrating GNN-Transformer prediction, Physics-Informed Neural Networks, 
    and Reinforcement Learning for optimal discharge control.
    """)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Control Panel")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 60, 600, 300)
    
    show_physics = st.sidebar.checkbox("Show physics constraints", value=True)
    show_rl_decision = st.sidebar.checkbox("Show RL decisions", value=True)
    
    # Main dashboard
    st.header("📊 Real-time Monitoring")
    
    # Layout: 3 columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌊 Water Levels")
        # Placeholder for real data
        stations = ["Station A", "Station B", "Station C", "Station D", "Station E"]
        levels = [52.3, 48.7, 55.1, 49.2, 51.8]
        
        fig = go.Figure()
        colors = ['green' if l < 55 else 'orange' if l < 65 else 'red' for l in levels]
        
        fig.add_trace(go.Bar(
            x=stations,
            y=levels,
            marker_color=colors,
            text=[f"{l:.1f}m" for l in levels],
            textposition='auto'
        ))
        
        fig.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="Warning")
        fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Critical")
        
        fig.update_layout(
            yaxis_title="Water Level (m)",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Model Confidence")
        
        # Confidence gauge
        confidence = 0.85
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Consensus Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 70], 'color': "red"},
                    {'range': [70, 85], 'color': "orange"},
                    {'range': [85, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("🎯 RL Decision")
        
        # Display RL recommendation
        st.metric(
            label="Recommended Discharge",
            value="2,450 m³/s",
            delta="+150 from current"
        )
        
        st.metric(
            label="Expected Water Level",
            value="51.2 m",
            delta="-0.8 m"
        )
        
        st.metric(
            label="Risk Level",
            value="MODERATE",
            delta=None
        )
    
    # Prediction visualization
    st.header("🔮 GNN-Transformer Predictions")
    
    # Time series prediction with confidence band
    hours_ahead = 6
    time_points = [datetime.now() + timedelta(hours=i) for i in range(hours_ahead)]
    
    # Synthetic prediction data
    base_level = 52.0
    prediction = base_level + np.sin(np.linspace(0, 2*np.pi, hours_ahead)) * 3 + np.random.randn(hours_ahead) * 0.5
    confidence_band = 2.0 + np.random.randn(hours_ahead) * 0.3
    
    fig = go.Figure()
    
    # Confidence band
    fig.add_trace(go.Scatter(
        x=time_points,
        y=prediction + confidence_band,
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=prediction - confidence_band,
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        mode='lines',
        line_color='rgba(0,100,80,0)',
        name='Confidence Band'
    ))
    
    # Prediction line
    fig.add_trace(go.Scatter(
        x=time_points,
        y=prediction,
        mode='lines+markers',
        line=dict(color='rgb(0,100,80)', width=2),
        name='Predicted Level'
    ))
    
    # Warning thresholds
    fig.add_hline(y=55, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Critical")
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Water Level (m)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Physics constraints monitoring
    if show_physics:
        st.header("⚙️ Physics Constraints (PINN)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Continuity Equation Residual")
            continuity_res = np.random.randn(10) * 0.05
            st.line_chart(continuity_res)
        
        with col2:
            st.subheader("Momentum Equation Residual")
            momentum_res = np.random.randn(10) * 0.08
            st.line_chart(momentum_res)
        
        # Physics compliance status
        avg_residual = np.mean(np.abs(continuity_res)) + np.mean(np.abs(momentum_res))
        if avg_residual < 0.1:
            st.success("✓ Physics constraints satisfied")
        else:
            st.warning("⚠️ Physics residuals elevated - manual review recommended")
    
    # Alert log
    st.header("🚨 Recent Alerts")
    
    alert_data = {
        'Time': [datetime.now() - timedelta(minutes=m) for m in [5, 15, 30, 45]],
        'Station': ['HYD_001', 'HYD_003', 'HYD_002', 'HYD_005'],
        'Level': [56.2, 58.1, 52.3, 54.7],
        'Alert Type': ['WARNING', 'WARNING', 'INFO', 'INFO'],
        'Status': ['Active', 'Resolved', 'Resolved', 'Active']
    }
    
    alert_df = pd.DataFrame(alert_data)
    
    # Color code by status
    def color_status(val):
        if val == 'Active':
            return 'background-color: red; color: white'
        elif val == 'Resolved':
            return 'background-color: green; color: white'
        return ''
    
    st.dataframe(
        alert_df.style.applymap(color_status, subset=['Status']),
        use_container_width=True
    )
    
    # Drone Vision Monitoring Section
    st.header("🚁 Drone Real-time Monitoring")
    
    col_drone1, col_drone2 = st.columns([2, 1])
    
    with col_drone1:
        st.subheader("Live Drone Feed")
        
        # Placeholder for drone image
        st.info("📡 Waiting for drone image stream...")
        
        # Simulate drone image upload
        uploaded_file = st.file_uploader(
            "Upload drone image for analysis",
            type=['jpg', 'jpeg', 'png'],
            key='drone_upload'
        )
        
        if uploaded_file is not None:
            from PIL import Image
            import io
            
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Drone Image - {datetime.now().strftime('%H:%M:%S')}", use_container_width=True)
            
            # Interactive click for SAM segmentation
            st.markdown("**🖱️ Click on image to segment suspicious regions**")
            st.markdown("*Click coordinates will be sent to SAM for segmentation*")
            
            # Mock click interface
            click_x = st.slider("Click X coordinate", 0, image.width, image.width // 2)
            click_y = st.slider("Click Y coordinate", 0, image.height, image.height // 2)
            
            if st.button("🔍 Analyze with SAM"):
                st.info("Sending to SAM for segmentation...")
                # Would call SAM API here
                st.success("✓ Segmentation complete - Crack region identified")
                
                # Mock segmentation overlay
                st.markdown("**Segmentation Result:**")
                st.markdown("- Crack detected: ~15cm width")
                st.markdown("- Location: Lower right quadrant")
                st.markdown("- Confidence: 87%")
    
    with col_drone2:
        st.subheader("AI Risk Assessment")
        
        # Vision analysis results
        if uploaded_file is not None:
            st.markdown("---")
            st.markdown("**🤖 Gemini AI Diagnosis:**")
            
            # Mock diagnosis
            diagnosis = {
                'visual_findings': {
                    'anomaly_type': 'crack',
                    'confidence': 0.87,
                    'anomaly_score': 0.72
                },
                'sensor_context': {
                    'water_level': 58.3,
                    'flow_velocity': 3.2,
                    'relative_to_warning': 0.33
                },
                'risk_assessment': 'high',
                'breach_probability': 0.68,
                'alert_triggered': True
            }
            
            # Risk indicator
            risk_color = 'red' if diagnosis['breach_probability'] > 0.7 else 'orange' if diagnosis['breach_probability'] > 0.5 else 'green'
            st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-size:18px;'>{diagnosis['risk_assessment'].upper()}</span>", unsafe_allow_html=True)
            
            # Breach probability gauge
            st.progress(diagnosis['breach_probability'], text=f"Breach Probability: {diagnosis['breach_probability']:.1%}")
            
            # Visual findings
            st.markdown("**Visual Analysis:**")
            st.markdown(f"- Anomaly Type: {diagnosis['visual_findings']['anomaly_type']}")
            st.markdown(f"- Confidence: {diagnosis['visual_findings']['confidence']:.1%}")
            
            # Sensor context
            st.markdown("**Sensor Context:**")
            st.markdown(f"- Water Level: {diagnosis['sensor_context']['water_level']:.1f}m")
            st.markdown(f"- Flow Velocity: {diagnosis['sensor_context']['flow_velocity']:.1f}m/s")
            
            # Alert status
            if diagnosis['alert_triggered']:
                st.error("🚨 **CRITICAL: Dike Breach Risk Detected!**")
                st.markdown("**Immediate Actions Required:**")
                st.markdown("1. ⚠️ Emergency inspection team dispatch")
                st.markdown("2. 📞 Alert downstream communities")
                st.markdown("3. 📊 Increase monitoring to every 15 minutes")
                st.markdown("4. 🚧 Prepare emergency sandbags")
                
                # GPS coordinates
                st.markdown("**📍 Location:**")
                st.markdown("- Latitude: 34.5241° N")
                st.markdown("- Longitude: 110.2837° E")
                st.markdown("- Dike Section: K12+350")
            else:
                st.success("✓ Risk within acceptable limits")
        else:
            st.info("Upload drone image to see AI analysis")
            
        # Vision system status
        st.markdown("---")
        st.markdown("**System Status:**")
        vision_status = {
            'ViT Model': '✓ Active',
            'SAM API': '⚠ Standby',
            'Gemini LLM': '✓ Connected',
            'Last Update': datetime.now().strftime('%H:%M:%S')
        }
        for key, val in vision_status.items():
            st.markdown(f"- {key}: {val}")
    
    # System status
    st.sidebar.header("📈 System Status")
    st.sidebar.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    st.sidebar.metric("Active Stations", "5/5")
    st.sidebar.metric("Model Latency", "42s", delta="-8s")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    # This runs when executed via `streamlit run dashboard.py`
    launch_monitoring_dashboard()
