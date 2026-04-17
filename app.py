"""Yellow River Digital Twin - Streamlit Interactive Demo.

交互式演示系统：对比 Baseline 与 PINN 模型的水沙预测能力。

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from pinn_model import YellowRiverPINN

# --- 页面配置 ---
st.set_page_config(
    page_title="Yellow River Digital Twin",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 黄河数字孪生：PINN 水沙预测演示系统")
st.markdown("""
本项目展示了**物理信息神经网络 (PINN)** 如何在预测黄河水位与泥沙时，
通过结合圣维南连续性方程，获得比传统 AI 更精准、更符合物理规律的结果。

**关键方程**: ∂h/∂t + α·∂(h·u)/∂x = 0
""")

# --- 侧边栏：输入参数 ---
st.sidebar.header("🕹️ 模拟参数控制")
st.sidebar.markdown("调整参数观察 PINN 的物理一致性优势")

t_slider = st.sidebar.slider(
    "时间点 (Time Index)", 
    min_value=0, 
    max_value=365, 
    value=180,
    help="模拟一年中的时间点 (0-365)"
)

rainfall_input = st.sidebar.slider(
    "模拟降雨强度 (Rainfall Intensity)", 
    min_value=0.0, 
    max_value=5.0, 
    value=1.2,
    step=0.1,
    help="单位: mm/h"
)

flow_input = st.sidebar.slider(
    "上游泄洪流量 (Upstream Discharge)", 
    min_value=500, 
    max_value=5000, 
    value=2000,
    step=100,
    help="单位: m³/s"
)

temperature = st.sidebar.slider(
    "气温 (Temperature)",
    min_value=-10.0,
    max_value=40.0,
    value=20.0,
    step=1.0,
    help="单位: °C"
)

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **PINN 优势**：
在极端流量条件下，PINN 模型通过物理约束保持水位曲线的连续性和平滑性，
避免纯数据驱动模型可能出现的非物理震荡。
""")

# --- 模型加载函数 ---
@st.cache_resource
def load_models():
    """加载 Baseline 和 PINN 模型。"""
    # 初始化模型
    model_b = YellowRiverPINN(input_dim=5, output_dim=2, activation='swish')
    model_p = YellowRiverPINN(input_dim=5, output_dim=2, activation='swish')
    
    # 模型路径
    baseline_path = Path("models/model_baseline.pth")
    pinn_path = Path("models/model_pinn.pth")
    
    models_loaded = False
    
    try:
        if baseline_path.exists() and pinn_path.exists():
            # 加载 Baseline 模型
            checkpoint_b = torch.load(
                baseline_path, 
                map_location=torch.device('cpu'),
                weights_only=False
            )
            model_b.load_state_dict(checkpoint_b['model_state_dict'])
            
            # 加载 PINN 模型
            checkpoint_p = torch.load(
                pinn_path, 
                map_location=torch.device('cpu'),
                weights_only=False
            )
            model_p.load_state_dict(checkpoint_p['model_state_dict'])
            
            models_loaded = True
        else:
            st.warning("⚠️ 未找到训练好的模型文件，请确保以下路径存在：")
            if not baseline_path.exists():
                st.warning(f"  - {baseline_path} (缺失)")
            if not pinn_path.exists():
                st.warning(f"  - {pinn_path} (缺失)")
            st.info("💡 提示：运行训练脚本生成模型文件")
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        st.info("💡 当前展示的是随机初始化状态")
    
    model_b.eval()
    model_p.eval()
    
    return model_b, model_p, models_loaded

# 加载模型
with st.spinner("加载模型中..."):
    model_baseline, model_pinn, models_loaded = load_models()

if models_loaded:
    st.success("✅ 模型加载成功！")
else:
    st.error("⚠️ 使用未训练的随机模型进行演示")

# --- 生成全河段推理数据 ---
x_range = np.linspace(0, 1, 100)  # 模拟河段位置 0 到 1 (0% 到 100%)

# 构造输入向量: [t, x, flow, rain, temp]
# 注意：这里使用归一化输入，与训练时保持一致
t_norm = t_slider / 365.0  # 时间归一化到 [0, 1]
flow_norm = flow_input / 5000.0  # 流量归一化到 [0, 1]
rain_norm = rainfall_input / 5.0  # 降雨归一化到 [0, 1]
temp_norm = (temperature + 10) / 50.0  # 温度归一化到 [0, 1]

inputs = []
for x in x_range:
    inputs.append([t_norm, x, flow_norm, rain_norm, temp_norm])

inputs_tensor = torch.FloatTensor(inputs)

# 模型推理
with torch.no_grad():
    pred_b = model_baseline(inputs_tensor).numpy()
    pred_p = model_pinn(inputs_tensor).numpy()

# 反归一化到原始尺度 (假设训练时的归一化参数)
# 这里使用近似值，实际应从 norm_params.json 加载
water_mean, water_std = 50.0, 20.0
sediment_mean, sediment_std = 10.0, 5.0

pred_b[:, 0] = pred_b[:, 0] * water_std + water_mean
pred_p[:, 0] = pred_p[:, 0] * water_std + water_mean
pred_b[:, 1] = pred_b[:, 1] * sediment_std + sediment_mean
pred_p[:, 1] = pred_p[:, 1] * sediment_std + sediment_mean

# --- 结果可视化 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 水位时空预测 (Water Level)")
    
    fig_h = go.Figure()
    
    fig_h.add_trace(go.Scatter(
        x=x_range * 100,  # 转换为百分比
        y=pred_b[:, 0], 
        name="Baseline (纯AI, λ=0.0)", 
        line=dict(dash='dash', color='gray', width=2),
        mode='lines'
    ))
    
    fig_h.add_trace(go.Scatter(
        x=x_range * 100,
        y=pred_p[:, 0], 
        name="PINN (物理增强, λ=0.1)", 
        line=dict(color='#0066CC', width=3),
        mode='lines'
    ))
    
    fig_h.update_layout(
        xaxis_title="河段位置 (km, 假设总长100km)", 
        yaxis_title="水位高度 (m)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_h, use_container_width=True)
    
    # 显示统计信息
    diff_h = np.abs(pred_p[:, 0] - pred_b[:, 0])
    st.caption(f"水位差异 | 最大: {diff_h.max():.2f}m | 平均: {diff_h.mean():.2f}m")

with col2:
    st.subheader("⏳ 含沙量分布预测 (Sediment Content)")
    
    fig_s = go.Figure()
    
    fig_s.add_trace(go.Scatter(
        x=x_range * 100,
        y=pred_b[:, 1], 
        name="Baseline (纯AI, λ=0.0)", 
        line=dict(dash='dash', color='gray', width=2),
        mode='lines'
    ))
    
    fig_s.add_trace(go.Scatter(
        x=x_range * 100,
        y=pred_p[:, 1], 
        name="PINN (物理增强, λ=0.1)", 
        line=dict(color='#FF6600', width=3),
        mode='lines'
    ))
    
    fig_s.update_layout(
        xaxis_title="河段位置 (km, 假设总长100km)", 
        yaxis_title="含沙量 (kg/m³)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_s, use_container_width=True)
    
    # 显示统计信息
    diff_s = np.abs(pred_p[:, 1] - pred_b[:, 1])
    st.caption(f"含沙量差异 | 最大: {diff_s.max():.2f}kg/m³ | 平均: {diff_s.mean():.2f}kg/m³")

# --- 物理残差计算 ---
st.markdown("---")
st.subheader("🔬 物理一致性分析")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric(
        label="当前时间点",
        value=f"第 {t_slider} 天"
    )

with col4:
    st.metric(
        label="上游流量",
        value=f"{flow_input} m³/s"
    )

with col5:
    st.metric(
        label="降雨强度",
        value=f"{rainfall_input} mm/h"
    )

st.info("""
**物理约束说明**：
PINN 模型在训练时优化了额外的物理损失项 `Physics_Loss = MSE(∂h/∂t + α·∂(h·u)/∂x, 0)`，
这确保了预测结果不仅拟合观测数据，还满足水力学连续性方程。

在极端降雨或高流量条件下，PINN 的预测曲线通常更平滑、更符合物理直觉。
""")

# --- 页脚 ---
st.markdown("---")
st.caption("🌊 Yellow River Digital Twin | Physics-Informed Neural Network Demo | 黄河数字孪生演示系统")
