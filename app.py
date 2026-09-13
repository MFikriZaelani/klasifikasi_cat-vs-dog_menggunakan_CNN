"""
Cats vs Dogs Classifier - Deep Learning CNN Application
Modern, Elegant, Clean & Professional UI built with Streamlit
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import io
import textwrap

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render clean HTML without markdown indentation issues
def render_html(html_str: str):
    st.markdown(textwrap.dedent(html_str).strip(), unsafe_allow_html=True)

# ==================== CUSTOM CSS (MODERN & CLEAN DESIGN SYSTEM) ====================
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #0F172A;
}

.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1180px !important;
}

.stApp {
    background-color: #F8FAFC;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #F0FDFA;
    color: #0D9488;
    border: 1px solid #CCFBF1;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.main-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 0.4rem 0;
}

.main-subtitle {
    font-size: 0.98rem;
    color: #64748B;
    font-weight: 400;
    margin: 0 0 1.5rem 0;
    line-height: 1.5;
}

.section-header-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 0.9rem 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
}

.section-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
}

.section-header-tag {
    font-size: 0.78rem;
    color: #0D9488;
    background: #F0FDFA;
    border: 1px solid #CCFBF1;
    padding: 3px 10px;
    border-radius: 9999px;
    font-weight: 600;
}

.prediction-hero-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    padding: 1.75rem 1.5rem;
    box-shadow: 0 10px 25px -4px rgba(13, 148, 136, 0.08), 0 4px 10px rgba(15, 23, 42, 0.03);
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}

.prediction-hero-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: #0D9488;
}

.icon-circle-badge {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.6rem;
    margin: 0.5rem auto 0.75rem auto;
    background: #F0FDFA;
    border: 2px solid #CCFBF1;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.12);
}

.pred-label-text {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
    line-height: 1.1;
}

.confidence-large {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0D9488;
    letter-spacing: -0.01em;
    margin: 0;
}

.confidence-subtext {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 1.25rem;
}

.status-badge-high {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F0FDF4;
    color: #16A34A;
    border: 1px solid #BBF7D0;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.status-badge-low {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFBEB;
    color: #D97706;
    border: 1px solid #FDE68A;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}

.chart-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1rem 1.15rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
    margin-bottom: 1rem;
}

.prob-bar-container {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-top: 1rem;
    text-align: left;
}

.prob-row {
    margin-bottom: 0.75rem;
}

.prob-row:last-child {
    margin-bottom: 0;
}

.prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.35rem;
}

.prob-track {
    width: 100%;
    height: 9px;
    background: #E2E8F0;
    border-radius: 9999px;
    overflow: hidden;
}

.prob-fill-teal {
    height: 100%;
    background: #0D9488;
    border-radius: 9999px;
}

.prob-fill-muted {
    height: 100%;
    background: #94A3B8;
    border-radius: 9999px;
}

.empty-state-box {
    background: #FFFFFF;
    border: 2px dashed #E2E8F0;
    border-radius: 20px;
    padding: 3rem 1.5rem;
    text-align: center;
    color: #64748B;
    margin-bottom: 1rem;
}

.empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 0.6rem;
    opacity: 0.75;
}

.empty-state-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.3rem;
}

.empty-state-desc {
    font-size: 0.85rem;
    color: #64748B;
    max-width: 320px;
    margin: 0 auto;
    line-height: 1.4;
}

.preview-frame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
    background: #F8FAFC;
    margin-bottom: 0.75rem;
}

.sidebar-section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #64748B;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.model-info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 0.75rem;
}

.model-info-pill {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 8px 10px;
}

.model-info-label {
    font-size: 0.68rem;
    color: #64748B;
    font-weight: 600;
    text-transform: uppercase;
    display: block;
    margin-bottom: 2px;
}

.model-info-val {
    font-size: 0.82rem;
    color: #0F172A;
    font-weight: 700;
    display: block;
}

.about-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.9rem;
    font-size: 0.8rem;
    line-height: 1.45;
    color: #475569;
}

.stButton > button {
    width: 100%;
    background-color: #0D9488 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.6rem 1.25rem !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #0F766E !important;
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.35) !important;
    transform: translateY(-1px);
}

.stDownloadButton > button {
    width: 100%;
    background-color: #FFFFFF !important;
    color: #0D9488 !important;
    border: 1.5px solid #0D9488 !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background-color: #F0FDFA !important;
    color: #0F766E !important;
    border-color: #0F766E !important;
}

.stat-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
}

.stat-val {
    font-size: 1.4rem;
    font-weight: 800;
    color: #0D9488;
    line-height: 1.1;
}

.stat-lbl {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    margin-top: 4px;
}

.batch-item-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""")

# ==================== LOAD MODEL ====================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load pre-trained CNN model"""
    try:
        model = keras.models.load_model('cats_vs_dogs_cnn_final.keras')
        return model
    except Exception as e:
        try:
            model = keras.models.load_model('best_model.h5')
            return model
        except Exception as e2:
            st.error(f"Gagal memuat model: {e}")
            st.info("Pastikan file 'cats_vs_dogs_cnn_final.keras' atau 'best_model.h5' berada di folder yang sama.")
            return None

# ==================== PREPROCESSING ====================
def preprocess_image(image, img_size=128):
    """Preprocess image untuk model inference CNN"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==================== PREDICTION ====================
def predict_image(model, image, img_size=128):
    """Eksekusi prediksi model terhadap gambar"""
    processed_img = preprocess_image(image, img_size)
    prediction = float(model.predict(processed_img, verbose=0)[0][0])
    
    if prediction > 0.5:
        label = "Dog"
        confidence = prediction
    else:
        label = "Cat"
        confidence = 1.0 - prediction
    
    probabilities = {
        'Cat': round((1.0 - prediction) * 100, 2),
        'Dog': round(prediction * 100, 2)
    }
    return label, confidence, probabilities

# ==================== DIAGRAMS & VISUALIZATIONS ====================
def create_gauge_chart(confidence, label, threshold=0.8):
    """Modern gauge indicator for confidence score"""
    val = confidence * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "<b>Confidence Meter</b>",
            'font': {'size': 15, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'}
        },
        number={
            'suffix': "%",
            'font': {'size': 32, 'color': '#0F172A', 'family': 'Plus Jakarta Sans', 'weight': 800},
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#94A3B8",
                'tickfont': {'size': 10, 'color': '#64748B', 'family': 'Plus Jakarta Sans'},
                'dtick': 25
            },
            'bar': {'color': "#0D9488", 'thickness': 0.65},
            'bgcolor': "#F1F5F9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#F8FAFC'},
                {'range': [50, threshold * 100], 'color': '#F1F5F9'},
                {'range': [threshold * 100, 100], 'color': '#E6FFFA'}
            ],
            'threshold': {
                'line': {'color': "#0F766E", 'width': 3},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=230,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Plus Jakarta Sans'}
    )
    return fig

def create_probability_bar(probabilities, label):
    """Modern horizontal bar chart for Cat vs Dog probabilities"""
    classes = ['Cat', 'Dog']
    values = [probabilities['Cat'], probabilities['Dog']]
    colors = ['#0D9488' if c == label else '#94A3B8' for c in classes]
    icons = ['🐱 Cat', '🐶 Dog']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=icons,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=8
        ),
        text=[f"<b>{v:.1f}%</b>" for v in values],
        textposition='auto',
        textfont=dict(size=13, color='#FFFFFF', family='Plus Jakarta Sans'),
        cliponaxis=False,
        hoverinfo='x+y'
    ))
    
    fig.update_layout(
        title={
            'text': "<b>Class Probabilities</b>",
            'font': {'size': 15, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'},
            'x': 0.0,
            'xanchor': 'left'
        },
        xaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor='#F1F5F9',
            zeroline=False,
            showline=False,
            ticksuffix="%",
            tickfont=dict(size=11, color='#64748B', family='Plus Jakarta Sans')
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            autorange="reversed",
            tickfont=dict(size=13, color='#0F172A', family='Plus Jakarta Sans', weight=600)
        ),
        height=190,
        margin=dict(l=10, r=15, t=35, b=15),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def create_batch_distribution_pie(cat_count, dog_count):
    """Modern Donut chart for batch class distribution"""
    labels = ['🐱 Cat', '🐶 Dog']
    values = [cat_count, dog_count]
    colors = ['#0D9488', '#334155']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color='#FFFFFF', family='Plus Jakarta Sans'),
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title={
            'text': "<b>Distribusi Kelas (Cat vs Dog)</b>",
            'font': {'size': 14, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'},
            'x': 0.0,
            'xanchor': 'left'
        },
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#475569", family='Plus Jakarta Sans')
        )
    )
    return fig

def create_batch_confidence_bar(results, threshold=0.8):
    """Modern Bar chart for confidence per image in batch"""
    filenames = [r['filename'] if len(r['filename']) <= 14 else r['filename'][:11] + '...' for r in results]
    full_filenames = [r['filename'] for r in results]
    conf_values = [r['confidence'] * 100 for r in results]
    labels = [r['label'] for r in results]
    colors = ['#0D9488' if c >= threshold * 100 else '#F59E0B' for c in conf_values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=filenames,
        y=conf_values,
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:.1f}%" for v in conf_values],
        textposition='auto',
        textfont=dict(size=11, family='Plus Jakarta Sans'),
        customdata=list(zip(full_filenames, labels)),
        hovertemplate="<b>%{customdata[0]}</b><br>Prediksi: %{customdata[1]}<br>Confidence: %{y:.1f}%<extra></extra>"
    ))
    
    # Add horizontal threshold line
    fig.add_hline(
        y=threshold * 100,
        line_dash="dot",
        line_color="#0F766E",
        line_width=2,
        annotation_text=f"Ambang ({int(threshold*100)}%)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#0F766E", family='Plus Jakarta Sans')
    )
    
    fig.update_layout(
        title={
            'text': "<b>Tingkat Confidence per Citra</b>",
            'font': {'size': 14, 'color': '#0F172A', 'family': 'Plus Jakarta Sans'},
            'x': 0.0,
            'xanchor': 'left'
        },
        yaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            showgrid=True,
            gridcolor='#F1F5F9',
            tickfont=dict(size=10, color='#64748B', family='Plus Jakarta Sans')
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, color='#64748B', family='Plus Jakarta Sans'),
            tickangle=-20
        ),
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# ==================== MAIN APP CONTROLLER ====================
def main():
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        render_html("""
        <div class="sidebar-section-title">Model Architecture</div>
        <div class="model-info-grid">
            <div class="model-info-pill">
                <span class="model-info-label">Network</span>
                <span class="model-info-val">CNN DeepNet</span>
            </div>
            <div class="model-info-pill">
                <span class="model-info-label">Input Size</span>
                <span class="model-info-val">128 × 128 px</span>
            </div>
            <div class="model-info-pill">
                <span class="model-info-label">Classes</span>
                <span class="model-info-val">Cat / Dog</span>
            </div>
            <div class="model-info-pill">
                <span class="model-info-label">Backend</span>
                <span class="model-info-val">Keras / TF</span>
            </div>
        </div>
        <div class="sidebar-section-title">Inference Settings</div>
        """)
        
        upload_mode = st.radio(
            "Upload Mode",
            ["Single Image", "Multiple Images"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.82rem; font-weight: 600; color: #475569;'>Confidence Threshold</span>", unsafe_allow_html=True)
        conf_threshold = st.slider(
            "Confidence Threshold Slider",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.05,
            format="%.2f",
            label_visibility="collapsed",
            help="Ambang batas minimal untuk kategori High Confidence."
        )
        
        render_html(f"<div style='font-size: 0.78rem; color: #64748B; margin-top: -6px; margin-bottom: 12px;'>Target ambang: <b>{int(conf_threshold*100)}%</b></div>")
        
        render_html("""
        <div class="sidebar-section-title">About Project</div>
        <div class="about-card">
            <p style="margin: 0 0 6px 0; font-weight: 700; color: #0F172A;">Cats & Dogs CNN Classifier</p>
            <p style="margin: 0 0 6px 0;">Model klasifikasi visi komputer CNN dengan regularisasi Dropout, Batch Normalization, dan Data Augmentation.</p>
            <div style="border-top: 1px solid #E2E8F0; padding-top: 6px; margin-top: 6px; font-size: 0.75rem;">
                <div>🎯 <b>Train Acc:</b> ~88.5%</div>
                <div>🧪 <b>Test Acc:</b> ~84.2%</div>
            </div>
        </div>
        <div style="height: 12px;'></div>
        <div style="text-align: center; font-size: 0.75rem; color: #94A3B8;">v1.2.0 • AI Vision Studio</div>
        """)

    # Main Header
    render_html("""
    <div class="hero-badge">⚡ Computer Vision Intelligence</div>
    <h1 class="main-title">🐾 Cats vs Dogs Classifier</h1>
    <p class="main-subtitle">Sistem klasifikasi citra berbasis Convolutional Neural Network (CNN) dengan analisis probabilitas mendalam.</p>
    """)

    # Views
    if upload_mode == "Single Image":
        render_single_image_view(model, conf_threshold)
    else:
        render_multiple_images_view(model, conf_threshold)

# ==================== SINGLE IMAGE VIEW ====================
def render_single_image_view(model, conf_threshold):
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        render_html("""
        <div class="section-header-card">
            <span class="section-header-title">📤 Upload Image</span>
            <span class="section-header-tag">JPG, JPEG, PNG</span>
        </div>
        """)
        
        uploaded_file = st.file_uploader(
            "Pilih foto kucing atau anjing",
            type=['jpg', 'jpeg', 'png'],
            help="Format file yang didukung: JPG, JPEG, PNG",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image_obj = Image.open(uploaded_file)
            st.image(image_obj, use_container_width=True)
            
            if st.button("🔮 Klasifikasikan Citra", key="btn_single_predict"):
                with st.spinner("Menganalisis fitur citra..."):
                    label, confidence, probabilities = predict_image(model, image_obj)
                    st.session_state.single_prediction = {
                        'label': label,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'filename': uploaded_file.name
                    }

    with col_result:
        render_html("""
        <div class="section-header-card">
            <span class="section-header-title">📊 Prediction Results</span>
            <span class="section-header-tag">CNN Inference</span>
        </div>
        """)
        
        if 'single_prediction' in st.session_state and uploaded_file is not None:
            pred = st.session_state.single_prediction
            label = pred['label']
            confidence = pred['confidence']
            probabilities = pred['probabilities']
            
            is_high_conf = confidence >= conf_threshold
            icon = "🐱" if label == "Cat" else "🐶"
            badge_class = "status-badge-high" if is_high_conf else "status-badge-low"
            badge_symbol = "●" if is_high_conf else "▲"
            badge_title = "High Confidence" if is_high_conf else "Low Confidence"
            
            cat_bar_class = "prob-fill-teal" if label == "Cat" else "prob-fill-muted"
            dog_bar_class = "prob-fill-teal" if label == "Dog" else "prob-fill-muted"
            
            # Focal Point Hero Card
            focal_html = f"""
            <div class="prediction-hero-card">
                <div style="display: flex; justify-content: flex-end; margin-bottom: 0.25rem;">
                    <div class="{badge_class}">
                        <span>{badge_symbol}</span>
                        <span>{badge_title} ({confidence*100:.1f}%)</span>
                    </div>
                </div>
                <div class="icon-circle-badge">
                    {icon}
                </div>
                <div class="pred-label-text">{label.upper()}</div>
                <div class="confidence-large">{confidence*100:.1f}%</div>
                <div class="confidence-subtext">Confidence Score</div>
                <div class="prob-bar-container">
                    <div class="prob-row">
                        <div class="prob-header">
                            <span>🐱 Cat Probability</span>
                            <span>{probabilities['Cat']:.1f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="{cat_bar_class}" style="width: {probabilities['Cat']}%;"></div>
                        </div>
                    </div>
                    <div class="prob-row">
                        <div class="prob-header">
                            <span>🐶 Dog Probability</span>
                            <span>{probabilities['Dog']:.1f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="{dog_bar_class}" style="width: {probabilities['Dog']}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """
            render_html(focal_html)
            
            # Interactive Visual Diagrams (Gauge & Bar Chart)
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(
                create_gauge_chart(confidence, label, conf_threshold),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(
                create_probability_bar(probabilities, label),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional Metric Details
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                render_html(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">MARGIN KEYAKINAN</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{abs(probabilities['Dog'] - probabilities['Cat']):.1f}%</div>
                </div>
                """)
            with info_col2:
                status_text = "Memenuhi Target" if is_high_conf else "Di Bawah Target"
                status_color = "#0D9488" if is_high_conf else "#D97706"
                render_html(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 10px 12px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #64748B; font-weight: 600;">STATUS AMBANG</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: {status_color};">{status_text}</div>
                </div>
                """)
        else:
            render_html("""
            <div class="empty-state-box">
                <div class="empty-state-icon">🖼️</div>
                <div class="empty-state-title">Menunggu Unggahan Gambar</div>
                <div class="empty-state-desc">Pilih atau letakkan gambar kucing atau anjing di panel sebelah kiri lalu tekan tombol klasifikasi.</div>
            </div>
            """)

# ==================== MULTIPLE IMAGES VIEW ====================
def render_multiple_images_view(model, conf_threshold):
    render_html("""
    <div class="section-header-card">
        <span class="section-header-title">📤 Batch Upload Images</span>
        <span class="section-header-tag">Multiple Files Supported</span>
    </div>
    """)
    
    uploaded_files = st.file_uploader(
        "Pilih beberapa gambar sekaligus",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Unggah koleksi foto kucing dan anjing untuk inferensi batch",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f"<div style='font-size: 0.88rem; color: #475569; margin-bottom: 10px;'>📁 <b>{len(uploaded_files)}</b> gambar siap diproses</div>", unsafe_allow_html=True)
        
        if st.button("🔮 Jalankan Prediksi Batch", key="btn_batch_predict"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.markdown(f"<span style='font-size: 0.85rem; color: #64748B;'>Memproses {idx + 1}/{len(uploaded_files)}: <b>{uploaded_file.name}</b>...</span>", unsafe_allow_html=True)
                
                image = Image.open(uploaded_file)
                label, confidence, probabilities = predict_image(model, image)
                
                results.append({
                    'filename': uploaded_file.name,
                    'image': image,
                    'label': label,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.batch_results = results
    
    if 'batch_results' in st.session_state and uploaded_files:
        results = st.session_state.batch_results
        
        cat_count = sum(1 for r in results if r['label'] == 'Cat')
        dog_count = sum(1 for r in results if r['label'] == 'Dog')
        avg_confidence = float(np.mean([r['confidence'] for r in results]))
        high_conf_count = sum(1 for r in results if r['confidence'] >= conf_threshold)
        
        # Summary KPI Cards
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            render_html(f"""
            <div class="stat-card">
                <div class="stat-val">{cat_count}</div>
                <div class="stat-lbl">🐱 Total Cats</div>
            </div>
            """)
        with kpi2:
            render_html(f"""
            <div class="stat-card">
                <div class="stat-val">{dog_count}</div>
                <div class="stat-lbl">🐶 Total Dogs</div>
            </div>
            """)
        with kpi3:
            render_html(f"""
            <div class="stat-card">
                <div class="stat-val">{avg_confidence*100:.1f}%</div>
                <div class="stat-lbl">📈 Rata-rata Confidence</div>
            </div>
            """)
        with kpi4:
            render_html(f"""
            <div class="stat-card">
                <div class="stat-val">{high_conf_count}/{len(results)}</div>
                <div class="stat-lbl">✅ High Confidence</div>
            </div>
            """)
            
        # Visual Batch Analytics Diagrams (Donut Chart & Confidence Chart)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        render_html("""
        <div class="section-header-card">
            <span class="section-header-title">📊 Analisis Visual Batch</span>
            <span class="section-header-tag">Batch Insights</span>
        </div>
        """)
        
        diag_col1, diag_col2 = st.columns([1, 1], gap="large")
        with diag_col1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(
                create_batch_distribution_pie(cat_count, dog_count),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with diag_col2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            st.plotly_chart(
                create_batch_confidence_bar(results, conf_threshold),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Individual Results Gallery
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_html(f"""
        <div class="section-header-card">
            <span class="section-header-title">🖼️ Hasil Inferensi Individual</span>
            <span class="section-header-tag">{len(results)} Citra Diproses</span>
        </div>
        """)
        
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            grid_cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(results):
                    res = results[idx]
                    with grid_cols[j]:
                        icon = "🐱" if res['label'] == 'Cat' else "🐶"
                        is_high = res['confidence'] >= conf_threshold
                        badge_style = "background: #F0FDF4; color: #16A34A; border: 1px solid #BBF7D0;" if is_high else "background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A;"
                        badge_text = "High" if is_high else "Low"
                        
                        render_html(f"""
                        <div class="batch-item-card">
                            <div style="font-size: 0.78rem; color: #64748B; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 6px; font-weight: 600;">
                                {res['filename']}
                            </div>
                        """)
                        st.image(res['image'], use_container_width=True)
                        render_html(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                                <span style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{icon} {res['label']}</span>
                                <span style="font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px; {badge_style}">{badge_text} ({res['confidence']*100:.1f}%)</span>
                            </div>
                            <div class="prob-track" style="height: 6px; margin-top: 6px;">
                                <div class="prob-fill-teal" style="width: {res['confidence']*100}%;"></div>
                            </div>
                        </div>
                        """)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_html("""
        <div class="section-header-card">
            <span class="section-header-title">💾 Ekspor Data Hasil Prediksi</span>
            <span class="section-header-tag">CSV Format</span>
        </div>
        """)
        
        df_results = pd.DataFrame([{
            'Filename': r['filename'],
            'Prediction': r['label'],
            'Confidence (%)': f"{r['confidence']*100:.2f}",
            'Cat Probability (%)': f"{r['probabilities']['Cat']:.2f}",
            'Dog Probability (%)': f"{r['probabilities']['Dog']:.2f}"
        } for r in results])
        
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Ringkasan Prediksi (CSV)",
            data=csv_data,
            file_name="cats_dogs_predictions.csv",
            mime="text/csv"
        )

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()