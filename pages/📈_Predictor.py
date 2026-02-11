
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import robust extractors for CPU/GPU series
from custom_transformers import CPUSeriesExtractor, GPUSeriesExtractor


# ── Page Config ──
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon=":chart_with_upwards_trend:",
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
    div[data-testid="stMetric"] label { color: #ffffff !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    button[data-testid="baseButton"] { background: #222222 !important; color: #ffffff !important; border: 1px solid #555555 !important; transition: background 0.2s, color 0.2s, border 0.2s, box-shadow 0.2s, transform 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
    button[data-testid="baseButton"]:hover { background: #444444 !important; color: #ffffff !important; border: 1px solid #bbbbbb !important; cursor: pointer; box-shadow: 0 8px 32px rgba(0,0,0,0.25); transform: translateY(-2px) scale(1.04); }
    button[data-testid="baseButton"]:active { background: #bbbbbb !important; color: #222222 !important; border: 1px solid #bbbbbb !important; box-shadow: 0 2px 8px rgba(0,0,0,0.10); transform: translateY(1px) scale(0.97); }
    h1, h2, h3, label, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp ul, .stApp ol, .stApp td, .stApp th { color: #ffffff !important; }
    hr { border-color: #555555 !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #222222, #181818); border-right: 1px solid #555555; }
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label { color: #ffffff !important; }
    .prediction-card { background: linear-gradient(135deg, #232323, #444444); border: 2px solid #888888; border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4); margin: 1rem 0; transition: box-shadow 0.2s, transform 0.2s; }
    .prediction-card:hover { box-shadow: 0 16px 48px rgba(0,0,0,0.6); transform: translateY(-6px) scale(1.04); z-index: 2; }
    .prediction-card:active { box-shadow: 0 2px 8px rgba(0,0,0,0.15); transform: translateY(2px) scale(0.97); }
    .prediction-price { font-size: 3.5rem; font-weight: bold; background: linear-gradient(90deg, #ffffff, #bbbbbb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .prediction-label { color: #ffffff !important; font-size: 1.1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──
@st.cache_data
def load_raw_data():
    return pd.read_csv("laptop_prices.csv")

raw_df = load_raw_data()

# Ensure raw_df also has CPU_series and GPU_series for lookups
if "CPU_series" not in raw_df.columns:
    raw_df = CPUSeriesExtractor().fit_transform(raw_df)
if "GPU_series" not in raw_df.columns:
    raw_df = GPUSeriesExtractor().fit_transform(raw_df)


# Use the robust extractors to add CPU_series and GPU_series columns for UI filtering
df = raw_df.copy()
df = CPUSeriesExtractor().fit_transform(df)
df = GPUSeriesExtractor().fit_transform(df)

# ── Model Loading ──
@st.cache_resource
def load_models():
    model_dir = "models"
    model_files = [
        ("XGBoost", "xgboost_pipe.joblib"),
        ("LightGBM", "lightgbm_pipe.joblib"),
        ("Random Forest Regressor", "random_forest_regressor_pipe.joblib"),
        ("Elastic Net", "elastic_net_pipe.joblib"),
        ("Linear Regression", "linear_regression_pipe.joblib")
    ]
    models = {}
    for name, fname in model_files:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models()

# ── Model Comparison Table ──
@st.cache_data
def load_comparison():
    path = os.path.join("models", "comparison.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return pd.DataFrame(columns=["Model", "  R2", " MAE", "RMSE"]) 

comparison = load_comparison()


# ── Hero Section ──
st.markdown("""
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 3rem; color: #ffffff; margin-bottom: 0.2rem;'>
    🤖 Price Predictor
    </h1>
    <p style='color: #bbbbbb; font-size: 1.1rem;'>
    Configure laptop specifications and get an instant ML-powered price estimate
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar: Model Selection & Comparison ──
with st.sidebar:
    st.markdown("## 🤖 Model Selection")
    
    selected_model_name = st.selectbox(
        ":trophy: Choose Model",
        options=list(models.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### :bar_chart: Model Comparison")
    
    comp_sorted = comparison.sort_values("  R2", ascending=True)
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(
        y=comp_sorted["Model"],
        x=comp_sorted["  R2"],
        orientation="h",
        marker=dict(
            color=comp_sorted["  R2"],
            colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#2ecc71"]],
            line=dict(color="#ffffff", width=0.5)
        ),
        text=comp_sorted["  R2"].round(3),
        textposition="inside",
        textfont=dict(color="white", size=12)
    ))
    fig_models.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccccee", size=11),
        height=250,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="R\u00b2 Score", range=[0, 1], gridcolor="#2a2a5a"),
        yaxis=dict(gridcolor="#2a2a5a")
    )
    st.plotly_chart(fig_models, use_container_width=True)
    
    # Show selected model metrics
    model_row_df = comparison[comparison["Model"] == selected_model_name]
    if not model_row_df.empty:
        model_row = model_row_df.iloc[0]
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1e1e3f, #2a2a5a); border: 1px solid #3a3a6a; 
        border-radius: 10px; padding: 12px; margin-top: 8px;'>
            <p style='color: #8888cc; margin: 0 0 6px 0; font-weight: bold;'>{selected_model_name} Metrics</p>
            <p style='color: #ffffff; margin: 2px 0;'>R\u00b2: <b>{model_row["  R2"]:.4f}</b></p>
            <p style='color: #ffffff; margin: 2px 0;'>MAE: <b>\u20ac{model_row[" MAE"]:.2f}</b></p>
            <p style='color: #ffffff; margin: 2px 0;'>RMSE: <b>\u20ac{model_row["RMSE"]:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"No metrics found for model: {selected_model_name}")

# ── Input Form ──
st.markdown("### :wrench: Configure Laptop Specifications")

st.markdown("### :bar_chart: Feature Importance")
try:
    selected_pipe = models[selected_model_name]
    feature_names = None
    importances = None
    # XGBoost, Random Forest, LightGBM, Elastic Net, Linear Regression
    if hasattr(selected_pipe, 'named_steps'):
        # Try XGBoost style
        if 'model' in selected_pipe.named_steps:
            model_step = selected_pipe.named_steps['model']
            # XGBoost
            if hasattr(model_step, 'regressor_') and hasattr(model_step.regressor_, 'feature_importances_'):
                xgb_core = model_step.regressor_
                if 'encoding_scaling' in selected_pipe.named_steps:
                    coltrans = selected_pipe.named_steps['encoding_scaling']
                    if 'cluster' in selected_pipe.named_steps:
                        cluster = selected_pipe.named_steps['cluster']
                        feature_names = cluster.get_feature_names_out(coltrans.get_feature_names_out())
                        importances = xgb_core.feature_importances_
            # Random Forest, LightGBM, etc.
            elif hasattr(model_step, 'feature_importances_'):
                importances = model_step.feature_importances_
                if 'encoding_scaling' in selected_pipe.named_steps:
                    coltrans = selected_pipe.named_steps['encoding_scaling']
                    feature_names = coltrans.get_feature_names_out()
        # Linear models
        elif 'regressor' in selected_pipe.named_steps:
            reg = selected_pipe.named_steps['regressor']
            if hasattr(reg, 'coef_'):
                importances = np.abs(reg.coef_)
                if 'encoding_scaling' in selected_pipe.named_steps:
                    coltrans = selected_pipe.named_steps['encoding_scaling']
                    feature_names = coltrans.get_feature_names_out()
    if feature_names is not None and importances is not None:
        # Clean feature names for display
        def clean_feature_name(name):
            return (name.replace('robust_', '')
                        .replace('std_', '')
                        .replace('ohe_', '')
                        .replace('binary_', '')
                        .replace('_', ' ')
                        .strip())
        # Group encoded columns by their original feature and average their importances
        cleaned_features = [clean_feature_name(f) for f in feature_names]
        feature_importance = pd.DataFrame({
            "feature": cleaned_features,
            "importance": importances
        })
        # Group by the base feature name (before first space or by splitting on space and taking the first part)
        def get_base_feature(name):
            # For features like "PrimaryStorage SSD", "PrimaryStorage HDD", etc.
            # Take the first word or up to the first space, or join until a number is found
            # If the feature name contains a space, take the part before the first space
            # But for features like "CPU freq" or "Screen Size", keep both words
            # We'll use a list of known multi-word features to avoid over-collapsing
            multi_word_features = [
                "Screen Size", "CPU freq", "Primary Storage", "Secondary Storage", "GPU company", "CPU company", "CPU series", "GPU series", "Retina Display", "IPS Panel", "Operating System", "Laptop Type"
            ]
            for mw in multi_word_features:
                if name.lower().startswith(mw.lower()):
                    return mw
            # Otherwise, take the first word
            return name.split()[0]

        feature_importance["base_feature"] = feature_importance["feature"].apply(get_base_feature)
        # Remove features containing 'te' (case-insensitive)
        filtered_importance = feature_importance[~feature_importance["base_feature"].str.lower().str.contains("te")]
        grouped_importance = filtered_importance.groupby("base_feature", as_index=False)["importance"].mean()
        grouped_importance = grouped_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)
        fig_feat = px.bar(
            grouped_importance,
            x="importance",
            y="base_feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Importance", "base_feature": "Feature"},
            height=400
        )
        fig_feat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccccee"),
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(gridcolor="#2a2a5a"),
            yaxis=dict(gridcolor="#2a2a5a")
        )
        fig_feat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")
except Exception as e:
    st.warning(f"Could not display feature importance: {str(e)}")

st.markdown("### :wrench: Configure Laptop Specifications")




col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", sorted(df["Company"].unique()))
    type_name = st.selectbox("Laptop Type", sorted(df["TypeName"].unique()))
    os = st.selectbox("Operating System", sorted(df["OS"].unique()))
    screen = st.selectbox("Screen Type", sorted(df["Screen"].unique()))
    inches_min = float(df["Inches"].min())
    inches_max = float(df["Inches"].max())
    inches = st.slider("Screen Size (Inches)", min_value=inches_min, max_value=inches_max, value=inches_min, step=0.1)
    weight_min = float(df["Weight"].min())
    weight_max = float(df["Weight"].max())
    weight = st.slider("Weight (KG)", min_value=weight_min, max_value=weight_max, value=weight_min, step=0.1)
    # Filter Pixels by selected Screen type
    if "Pixels" in df.columns:
        filtered_pixels = df[df["Screen"] == screen]["Pixels"]
        if not filtered_pixels.empty:
            pixel_values = sorted(filtered_pixels.unique())
        else:
            pixel_values = sorted(df["Pixels"].unique())
        if len(pixel_values) > 1:
            pixels_min = int(min(pixel_values))
            pixels_max = int(max(pixel_values))
            # Try to infer step from the most common difference
            if len(pixel_values) > 2:
                steps = [pixel_values[i+1] - pixel_values[i] for i in range(len(pixel_values)-1)]
                from collections import Counter
                step = Counter(steps).most_common(1)[0][0]
            else:
                step = pixel_values[1] - pixel_values[0]
            pixels = st.number_input("Pixels", min_value=pixels_min, max_value=pixels_max, value=pixels_min, step=step)
        else:
            pixels = st.number_input("Pixels", value=int(pixel_values[0]), disabled=True)
    else:
        # Try to get Pixels values from the data if possible
        if "Screen" in df.columns and "ScreenW" in df.columns and "ScreenH" in df.columns:
            filtered = df[df["Screen"] == screen]
            pixel_values = sorted((filtered["ScreenW"] * filtered["ScreenH"]).unique()) if not filtered.empty else []
            if not pixel_values:
                # fallback to all available pixel values in data
                pixel_values = sorted((df["ScreenW"] * df["ScreenH"]).unique())
            if pixel_values:
                pixels = st.selectbox("Pixels", pixel_values, index=0)
            else:
                pixels = st.number_input("Pixels", value=2073600)
        else:
            pixels = st.number_input("Pixels", value=2073600)
    # Move touchscreen, ips_panel, retina_display to col1 for balance
    touchscreen = st.selectbox("Touchscreen", sorted(df["Touchscreen"].unique()))
    ips_panel = st.selectbox("IPS Panel", sorted(df["IPSpanel"].unique()))
    retina_display = st.selectbox("Retina Display", sorted(df["RetinaDisplay"].unique()))

with col2:
    # RAM stepper with known intervals
    ram_options = sorted(df["Ram"].unique())
    if len(ram_options) > 1:
        ram_steps = [ram_options[i+1] - ram_options[i] for i in range(len(ram_options)-1)]
        from collections import Counter
        ram_step = Counter(ram_steps).most_common(1)[0][0]
    else:
        ram_step = 1
    ram = st.number_input("RAM (GB)", min_value=min(ram_options), max_value=max(ram_options), value=min(ram_options), step=ram_step)

    cpu_company = st.selectbox("CPU Company", sorted(df["CPU_company"].unique()))
    if "CPU_series" in df.columns:
        cpu_series_filtered = df[df["CPU_company"] == cpu_company]["CPU_series"].unique()
        if len(cpu_series_filtered) > 0:
            cpu_series = st.selectbox("CPU Series", sorted(cpu_series_filtered))
        else:
            cpu_series = st.selectbox("CPU Series", ["Unknown"], disabled=True)
    else:
        cpu_series = st.selectbox("CPU Series", ["Unknown"], disabled=True)
    # CPU Frequency slider with known intervals
    cpu_freq_filtered = sorted(df[df["CPU_company"] == cpu_company]["CPU_freq"].unique())
    if len(cpu_freq_filtered) > 1:
        cpu_freq_steps = [cpu_freq_filtered[i+1] - cpu_freq_filtered[i] for i in range(len(cpu_freq_filtered)-1)]
        from collections import Counter
        cpu_freq_step = Counter(cpu_freq_steps).most_common(1)[0][0]
        cpu_freq_min = float(min(cpu_freq_filtered))
        cpu_freq_max = float(max(cpu_freq_filtered))
        cpu_freq = st.slider("CPU Frequency (GHz)", min_value=cpu_freq_min, max_value=cpu_freq_max, value=cpu_freq_min, step=cpu_freq_step)
    elif len(cpu_freq_filtered) == 1:
        cpu_freq = st.number_input("CPU Frequency (GHz)", value=float(cpu_freq_filtered[0]), disabled=True)
    else:
        cpu_freq = st.number_input("CPU Frequency (GHz)", value=1.0, disabled=True)
    gpu_company = st.selectbox("GPU Company", sorted(df["GPU_company"].unique()))
    if "GPU_series" in df.columns:
        # Only show GPU_series that actually exist for the selected GPU_company
        filtered_gpu = df[df["GPU_company"] == gpu_company]
        gpu_series_options = filtered_gpu["GPU_series"].dropna().unique() if not filtered_gpu.empty else []
        if len(gpu_series_options) > 0:
            gpu_series = st.selectbox("GPU Series", sorted(gpu_series_options))
        else:
            gpu_series = st.selectbox("GPU Series", ["Unknown"], disabled=True)
    else:
        gpu_series = st.selectbox("GPU Series", ["Unknown"], disabled=True)
    primary_storage_type = st.selectbox("Primary Storage Type", sorted(df["PrimaryStorageType"].unique()))
    primary_storage_filtered = sorted(df[df["PrimaryStorageType"] == primary_storage_type]["PrimaryStorage"].unique())
    primary_storage_options = [f"{int(val)} GB" for val in primary_storage_filtered]
    primary_storage_map = dict(zip(primary_storage_options, primary_storage_filtered))
    primary_storage_label = st.selectbox("Primary Storage Size", primary_storage_options)
    primary_storage = primary_storage_map[primary_storage_label]
    secondary_storage_type = st.selectbox("Secondary Storage Type", sorted(df["SecondaryStorageType"].unique()))
    secondary_storage_filtered = sorted(df[df["SecondaryStorageType"] == secondary_storage_type]["SecondaryStorage"].unique())
    secondary_storage_options = [f"{int(val)} GB" for val in secondary_storage_filtered]
    secondary_storage_map = dict(zip(secondary_storage_options, secondary_storage_filtered))
    secondary_storage_label = st.selectbox("Secondary Storage Size", secondary_storage_options)
    secondary_storage = secondary_storage_map[secondary_storage_label]

st.divider()

# ── Prediction ──
if st.button(":crystal_ball: Predict Price", use_container_width=True):
    # Auto-select most common ScreenW and ScreenH for selected screen type
    screenw_default = raw_df[raw_df["Screen"] == screen]["ScreenW"].mode()
    screenw_default = screenw_default[0] if not screenw_default.empty else 1920
    screenh_default = raw_df[raw_df["Screen"] == screen]["ScreenH"].mode()
    screenh_default = screenh_default[0] if not screenh_default.empty else 1080
    # Auto-select most common CPU_model and GPU_model for selected series/company using raw_df
    cpu_model_default = raw_df[(raw_df["CPU_company"] == cpu_company) & (raw_df["CPU_series"] == cpu_series)]["CPU_model"].mode()
    cpu_model_default = cpu_model_default[0] if not cpu_model_default.empty else "Unknown"
    gpu_model_default = raw_df[(raw_df["GPU_company"] == gpu_company) & (raw_df["GPU_series"] == gpu_series)]["GPU_model"].mode()
    gpu_model_default = gpu_model_default[0] if not gpu_model_default.empty else "Unknown"

    input_data = pd.DataFrame([{
        "Company": company,
        "TypeName": type_name,
        "Inches": inches,
        "Ram": ram,
        "OS": os,
        "Weight": weight,
        "Screen": screen,
        "ScreenW": screenw_default,
        "ScreenH": screenh_default,
        "Touchscreen": touchscreen,
        "IPSpanel": ips_panel,
        "RetinaDisplay": retina_display,
        "CPU_company": cpu_company,
        "CPU_model": cpu_model_default,
        "CPU_freq": cpu_freq,
        "PrimaryStorage": primary_storage,
        "SecondaryStorage": secondary_storage,
        "PrimaryStorageType": primary_storage_type,
        "SecondaryStorageType": secondary_storage_type,
        "GPU_company": gpu_company,
        "GPU_model": gpu_model_default,
        "CPU_series": cpu_series,
        "GPU_series": gpu_series,
        "Pixels": pixels
    }])

    selected_pipe = models[selected_model_name]
    try:
        prediction = selected_pipe.predict(input_data)[0]
        # ...existing code...
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
    
    if selected_model_name not in models:
        st.error(f"Model '{selected_model_name}' is not available. Please select another model.")
    else:
        selected_pipe = models[selected_model_name]
    
    try:
        prediction = selected_pipe.predict(input_data)[0]
        
        # ── Prediction Result Card ──
        st.markdown(f"""
        <div class='prediction-card'>
            <div class='prediction-label'>Estimated Laptop Price ({selected_model_name})</div>
            <div class='prediction-price'>\u20ac{prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Context Metrics ──
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        avg_price = df["Price_euros"].mean()
        median_price = df["Price_euros"].median()
        diff_from_avg = ((prediction - avg_price) / avg_price) * 100
        
        # Find percentile
        percentile = (df["Price_euros"] < prediction).mean() * 100
        
        with col_m1:
            st.metric(":dart: Predicted", f"\u20ac{prediction:,.0f}")
        with col_m2:
            st.metric(":chart_with_upwards_trend: vs Average", 
                      f"{diff_from_avg:+.1f}%",
                      delta=f"\u20ac{prediction - avg_price:+,.0f}")
        with col_m3:
            st.metric(":100: Percentile", f"{percentile:.0f}th")
        with col_m4:
            # Price tier
            if prediction < df["Price_euros"].quantile(0.25):
                tier = "Budget"
            elif prediction < df["Price_euros"].quantile(0.50):
                tier = "Mid-range"
            elif prediction < df["Price_euros"].quantile(0.75):
                tier = "Premium"
            else:
                tier = "High-end"
            st.metric(":gem: Price Tier", tier)
        
        st.divider()
        
        # ── Row: Price in Context + Similar Laptops ──
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("### :moneybag: Price in Market Context")
            fig_context = px.histogram(
                df, x="Price_euros", nbins=50,
                color_discrete_sequence=["#667eea"],
                opacity=0.6,
                labels={"Price_euros": "Price (\u20ac)", "count": "Count"}
            )
            # Offset annotation positions if prediction and avg_price are close
            offset = (fig_context.data[0].x[-1] - fig_context.data[0].x[0]) * 0.01 if hasattr(fig_context.data[0], 'x') and fig_context.data[0].x is not None and len(fig_context.data[0].x) > 1 else 50
            pred_annot_y = 1.08
            avg_annot_y = 1.16
            if abs(prediction - avg_price) < offset * 2:
                pred_annot_y = 1.08
                avg_annot_y = 1.16
            else:
                pred_annot_y = avg_annot_y = 1.08
            fig_context.add_vline(
                x=prediction, line_dash="solid", line_color="#2ecc71", line_width=3,
                annotation_text=f"Your Laptop: \u20ac{prediction:,.0f}",
                annotation_font_color="#2ecc71",
                annotation_font_size=14,
                annotation_position="top left",
                annotation_y=pred_annot_y
            )
            fig_context.add_vline(
                x=avg_price, line_dash="dash", line_color="#ff6b6b",
                annotation_text=f"Market Avg: \u20ac{avg_price:,.0f}",
                annotation_font_color="#ff6b6b",
                annotation_position="top right",
                annotation_y=avg_annot_y
            )
            plot_font = "#ccccee"
            plot_xgrid = plot_ygrid = "#2a2a5a"
            plot_template = "plotly_dark"
            fig_context.update_layout(
                template=plot_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=plot_font),
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(gridcolor=plot_xgrid),
                yaxis=dict(gridcolor=plot_ygrid)
            )
            st.plotly_chart(fig_context, use_container_width=True)
        
        with col_viz2:
            st.markdown("### :mag: Similar Laptops in Dataset")
            # Find similar laptops by type, company, and RAM
            similar = df[
                (df["TypeName"] == type_name) &
                (df["Ram"] == ram)
            ].sort_values("Price_euros")
            
            if len(similar) == 0:
                similar = df[df["TypeName"] == type_name].sort_values("Price_euros")
            
            if len(similar) > 0:
                fig_similar = px.strip(
                    similar.head(50), y="Price_euros", x="Company",
                    color="Company",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"Price_euros": "Price (\u20ac)", "Company": ""}
                )
                fig_similar.add_hline(
                    y=prediction, line_dash="solid", line_color="#2ecc71", line_width=2,
                    annotation_text=f"Your prediction: \u20ac{prediction:,.0f}",
                    annotation_font_color="#2ecc71"
                )
                fig_similar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ccccee"),
                    height=400,
                    margin=dict(l=40, r=20, t=20, b=40),
                    showlegend=False,
                    xaxis=dict(gridcolor="#2a2a5a"),
                    yaxis=dict(gridcolor="#2a2a5a")
                )
                st.plotly_chart(fig_similar, use_container_width=True)
            else:
                st.info("No similar laptops found in the dataset.")
        
        # ── All Models Comparison ──
        st.markdown("### :robot_face: All Model Predictions")
        all_predictions = {}
        for name, pipe in models.items():
            try:
                pred = pipe.predict(input_data)[0]
                all_predictions[name] = pred
            except Exception:
                pass
        
        if all_predictions:
            pred_df = pd.DataFrame({
                "Model": list(all_predictions.keys()),
                "Prediction": list(all_predictions.values())
            }).sort_values("Prediction", ascending=True)
            
            fig_all = go.Figure()
            colors = ["#2ecc71" if m == selected_model_name else "#667eea" 
                      for m in pred_df["Model"]]
            
            fig_all.add_trace(go.Bar(
                y=pred_df["Model"],
                x=pred_df["Prediction"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#ffffff", width=0.5)),
                text=[f"\u20ac{p:,.0f}" for p in pred_df["Prediction"]],
                textposition="inside",
                textfont=dict(color="white", size=14)
            ))
            fig_all.update_layout(
                template=plot_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=plot_font),
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(title="Predicted Price (\u20ac)", gridcolor=plot_xgrid),
                yaxis=dict(gridcolor=plot_ygrid)
            )
            st.plotly_chart(fig_all, use_container_width=True)
            
            # Stats
            pred_values = list(all_predictions.values())
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric(":chart_with_downwards_trend: Lowest Estimate", 
                          f"\u20ac{min(pred_values):,.0f}")
            with col_s2:
                st.metric(":left_right_arrow: Average Estimate", 
                          f"\u20ac{np.mean(pred_values):,.0f}")
            with col_s3:
                st.metric(":chart_with_upwards_trend: Highest Estimate", 
                          f"\u20ac{max(pred_values):,.0f}")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# ── Footer ──
st.divider()
st.markdown("""
<div style='text-align: center; padding: 1rem 0; color: #5555aa;'>
    <p style='color: #bbbbbb;'>Built with Streamlit & Scikit-Learn | 
    Powered by XGBoost, LightGBM, Random Forest, Elastic Net & Linear Regression</p>
</div>
""", unsafe_allow_html=True)