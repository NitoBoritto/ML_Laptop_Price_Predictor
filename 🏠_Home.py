import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# ── Page Config ──
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# Always apply dark mode CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #181818 0%, #222222 50%, #181818 100%); }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #232323, #444444); border: 1px solid #555555; border-radius: 12px; padding: 16px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); transition: box-shadow 0.2s, transform 0.2s; }
    div[data-testid="stMetric"]:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.5); transform: translateY(-4px) scale(1.03); z-index: 2; }
    div[data-testid="stMetric"]:active { box-shadow: 0 2px 8px rgba(0,0,0,0.2); transform: translateY(1px) scale(0.98); }
    div[data-testid="stMetric"] label { color: #bbbbbb !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    button[data-testid="baseButton"] { background: #222222 !important; color: #ffffff !important; border: 1px solid #555555 !important; transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s, transform 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    button[data-testid="baseButton"]:hover { background: #444444 !important; color: #ffffff !important; border: 1px solid #bbbbbb !important; cursor: pointer; box-shadow: 0 8px 32px rgba(0,0,0,0.25); transform: translateY(-2px) scale(1.04); }
    button[data-testid="baseButton"]:active { background: #bbbbbb !important; color: #222222 !important; border: 1px solid #bbbbbb !important; box-shadow: 0 2px 8px rgba(0,0,0,0.10); transform: translateY(1px) scale(0.97); }
    h1, h2, h3, label, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp ul, .stApp ol, .stApp td, .stApp th { color: #ffffff !important; }
    hr { border-color: #555555 !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #222222, #181818); border-right: 1px solid #555555; }
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label { color: #bbbbbb !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
@st.cache_data
def load_data():
    return pd.read_csv("laptop_prices.csv")

df = load_data()

# ── Sidebar Filters ──
with st.sidebar:
    st.markdown("## :mag: Filters")
    
    # Brand filter
    all_brands = sorted(df["Company"].unique().tolist())
    selected_brands = st.multiselect(
        ":office: Brand",
        options=all_brands,
        default=[]
    )
    
    # Laptop type filter
    all_types = sorted(df["TypeName"].unique().tolist())
    selected_types = st.multiselect(
        ":package: Laptop Type",
        options=all_types,
        default=[]
    )
    
    # OS filter
    all_os = sorted(df["OS"].unique().tolist())
    selected_os = st.multiselect(
        ":desktop_computer: Operating System",
        options=all_os,
        default=[]
    )
    
    st.markdown("---")
    
    # Price range slider
    min_price = int(df["Price_euros"].min())
    max_price = int(df["Price_euros"].max())
    price_range = st.slider(
        ":moneybag: Price Range (EUR)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=50
    )
    
    # RAM filter
    all_ram = sorted(df["Ram"].unique().tolist())
    selected_ram = st.multiselect(
        ":brain: RAM (GB)",
        options=all_ram,
        default=[]
    )
    
    # CPU company filter
    all_cpu = sorted(df["CPU_company"].unique().tolist())
    selected_cpu = st.multiselect(
        ":zap: CPU Company",
        options=all_cpu,
        default=[]
    )
    
    st.markdown("---")
    st.markdown(
        "<p style='color:#5555aa; font-size:0.85rem; text-align:center;'>"
        "Showing filtered results</p>",
        unsafe_allow_html=True
    )

# ── Apply Filters (empty = show all) ──
filtered = df[
    (df["Company"].isin(selected_brands) if selected_brands else True) &
    (df["TypeName"].isin(selected_types) if selected_types else True) &
    (df["OS"].isin(selected_os) if selected_os else True) &
    (df["Ram"].isin(selected_ram) if selected_ram else True) &
    (df["CPU_company"].isin(selected_cpu) if selected_cpu else True) &
    (df["Price_euros"] >= price_range[0]) &
    (df["Price_euros"] <= price_range[1])
]


# ── Hero Section ──
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 3rem; color: #ffffff; margin-bottom: 0.2rem;'>
    💻 Laptop Price Predictor
    </h1>
    <p style='color: #bbbbbb; font-size: 1.1rem;'>
    Machine Learning powered price estimation across 1,200+ laptops
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── KPI Metrics ──
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(":bar_chart: Total Laptops", f"{len(filtered):,}")
with col2:
    st.metric(":office: Brands", filtered["Company"].nunique())
with col3:
    avg = filtered["Price_euros"].mean() if len(filtered) > 0 else 0
    st.metric(":money_with_wings: Avg Price", f"\u20ac{avg:,.0f}")
with col4:
    if len(filtered) > 0:
        lo = filtered["Price_euros"].min()
        hi = filtered["Price_euros"].max()
        st.metric(":chart_with_upwards_trend: Price Range", f"\u20ac{lo:,.0f} \u2013 \u20ac{hi:,.0f}")
    else:
        st.metric(":chart_with_upwards_trend: Price Range", "N/A")
with col5:
    st.metric(":computer: Filtered / Total", f"{len(filtered):,} / {len(df):,}")

st.divider()

# ── Guard for empty filter ──
if len(filtered) == 0:
    st.warning("No laptops match the current filters. Adjust the sidebar filters.")
    st.stop()

# ── Row 1: Price Distribution + Brand Market Share ──
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### :moneybag: Price Distribution")
    fig_price = px.histogram(
        filtered, x="Price_euros", nbins=50,
        color_discrete_sequence=["#667eea"],
        opacity=0.85,
        labels={"Price_euros": "Price (\u20ac)", "count": "Count"}
    )
    median_price = filtered["Price_euros"].median()
    fig_price.add_vline(
        x=median_price, line_dash="dash", line_color="#ff6b6b",
        annotation_text=f"Median: \u20ac{median_price:,.0f}",
        annotation_font_color="#ff6b6b"
    )
    plot_font = "#ccccee"
    plot_xgrid = plot_ygrid = "#2a2a5a"
    plot_template = "plotly_dark"
    fig_price.update_layout(
        template=plot_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=plot_font),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor=plot_xgrid),
        yaxis=dict(gridcolor=plot_ygrid)
    )
    st.plotly_chart(fig_price, use_container_width=True)

with col_right:
    st.markdown("### :office: Top Brands by Volume")
    top_brands = filtered["Company"].value_counts().head(10).reset_index()
    top_brands.columns = ["Company", "Count"]
    
    fig_brands = px.bar(
        top_brands, x="Count", y="Company", orientation="h",
        color="Count",
        color_continuous_scale=["#3a3a8a", "#667eea", "#764ba2"],
        labels={"Count": "Number of Laptops", "Company": ""}
    )
    fig_brands.update_layout(
        template=plot_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=plot_font),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", gridcolor=plot_ygrid),
        xaxis=dict(gridcolor=plot_xgrid)
    )
    st.plotly_chart(fig_brands, use_container_width=True)

# ── Row 2: Avg Price by Type + RAM vs Price ──
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("### :package: Average Price by Laptop Type")
    type_price = filtered.groupby("TypeName")["Price_euros"].mean().sort_values(ascending=True).reset_index()
    type_price.columns = ["Type", "Avg Price"]
    
    fig_type = px.bar(
        type_price, x="Avg Price", y="Type", orientation="h",
        color="Avg Price",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        labels={"Avg Price": "Average Price (\u20ac)", "Type": ""}
    )
    fig_type.update_layout(
        template=plot_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=plot_font),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor=plot_xgrid),
        yaxis=dict(gridcolor=plot_ygrid)
    )
    st.plotly_chart(fig_type, use_container_width=True)

with col_right2:
    st.markdown("### :brain: RAM vs Price")
    fig_ram = px.box(
        filtered, x="Ram", y="Price_euros",
        color_discrete_sequence=["#667eea"],
        labels={"Ram": "RAM (GB)", "Price_euros": "Price (\u20ac)"}
    )
    fig_ram.update_layout(
        template=plot_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=plot_font),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor=plot_xgrid),
        yaxis=dict(gridcolor=plot_ygrid)
    )
    st.plotly_chart(fig_ram, use_container_width=True)

# ── Footer ──
st.divider()
st.markdown("""
<div style='text-align: center; padding: 1rem 0; color: #5555aa;'>
    <p style='color: #bbbbbb;'>Built with Streamlit & Scikit-Learn | 
    Navigate to <b>Predictor</b> page to estimate laptop prices</p>
</div>
""", unsafe_allow_html=True)