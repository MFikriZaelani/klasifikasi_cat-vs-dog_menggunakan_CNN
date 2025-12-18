"""
Streamlit App untuk Klasifikasi Cats vs Dogs menggunakan CNN
Deploy model yang sudah di-training di Kaggle
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
import io

# ==================== KONFIGURASI PAGE ====================
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .cat-prediction {
        background: linear-gradient(135deg, #FFA07A, #FFB6C1);
        color: white;
    }
    .dog-prediction {
        background: linear-gradient(135deg, #87CEEB, #98D8C8);
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load pre-trained model"""
    try:
        # Ganti dengan path model Anda
        model = keras.models.load_model('cats_vs_dogs_cnn_final.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Pastikan file 'cats_vs_dogs_cnn_final.keras' ada di folder yang sama dengan app.py")
        return None

# ==================== PREPROCESSING ====================
def preprocess_image(image, img_size=128):
    """
    Preprocess image untuk prediksi
    Args:
        image: PIL Image atau numpy array
        img_size: ukuran target (default 128x128)
    Returns:
        preprocessed image array
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

# ==================== PREDICTION ====================
def predict_image(model, image, img_size=128):
    """
    Prediksi gambar
    Returns:
        label (str): 'Cat' atau 'Dog'
        confidence (float): confidence score
        probabilities (dict): probabilitas untuk setiap kelas
    """
    # Preprocess
    processed_img = preprocess_image(image, img_size)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Hasil
    if prediction > 0.5:
        label = "Dog"
        confidence = prediction
    else:
        label = "Cat"
        confidence = 1 - prediction
    
    probabilities = {
        'Cat': (1 - prediction) * 100,
        'Dog': prediction * 100
    }
    
    return label, confidence, probabilities

# ==================== VISUALIZATIONS ====================
def create_gauge_chart(confidence, label):
    """Create gauge chart untuk confidence"""
    color = "#FF6B6B" if label == "Cat" else "#4ECDC4"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Score", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 75], 'color': '#FFD9B3'},
                {'range': [75, 100], 'color': '#B3FFB3'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_probability_bar(probabilities):
    """Create bar chart untuk probabilitas"""
    fig = px.bar(
        x=list(probabilities.values()),
        y=list(probabilities.keys()),
        orientation='h',
        text=[f'{v:.2f}%' for v in probabilities.values()],
        color=list(probabilities.keys()),
        color_discrete_map={'Cat': '#FF6B6B', 'Dog': '#4ECDC4'}
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Class",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🐱 Cats vs Dogs Classifier 🐶</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload gambar kucing atau anjing untuk klasifikasi menggunakan Deep Learning CNN</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.info(f"""
        **Model Information:**
        - Architecture: CNN
        - Input Size: 128x128
        - Classes: Cat, Dog
        - Framework: TensorFlow/Keras
        """)
        
        # Upload mode
        upload_mode = st.radio(
            "Upload Mode:",
            ["Single Image", "Multiple Images"]
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum confidence untuk prediksi yang valid"
        )
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** 
        yang di-training pada dataset Cats and Dogs dari Kaggle.
        
        **Teknik yang digunakan:**
        - Data Augmentation
        - Batch Normalization
        - Dropout Layers
        - L2 Regularization
        
        **Performance:**
        - Train Accuracy: ~85-90%
        - Test Accuracy: ~80-85%
        """)
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")
    
    # Main content
    if upload_mode == "Single Image":
        single_image_mode(model, conf_threshold)
    else:
        multiple_images_mode(model, conf_threshold)

# ==================== SINGLE IMAGE MODE ====================
def single_image_mode(model, conf_threshold):
    """Mode untuk upload satu gambar"""
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload gambar kucing atau anjing"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Predict button
            if st.button("🔮 Predict", type="primary"):
                with st.spinner("Analyzing image..."):
                    label, confidence, probabilities = predict_image(model, image)
                    
                    # Store in session state
                    st.session_state.prediction = {
                        'label': label,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            label = pred['label']
            confidence = pred['confidence']
            probabilities = pred['probabilities']
            
            # Prediction box
            box_class = "cat-prediction" if label == "Cat" else "dog-prediction"
            emoji = "🐱" if label == "Cat" else "🐶"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h1>{emoji} This is a {label}! {emoji}</h1>
                <h2>Confidence: {confidence*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if confidence >= conf_threshold:
                st.success(f"✅ High confidence prediction (≥ {conf_threshold*100:.0f}%)")
            else:
                st.warning(f"⚠️ Low confidence prediction (< {conf_threshold*100:.0f}%)")
            
            # Gauge chart
            st.plotly_chart(
                create_gauge_chart(confidence, label),
                use_container_width=True
            )
            
            # Probability bar chart
            st.plotly_chart(
                create_probability_bar(probabilities),
                use_container_width=True
            )
            
            # Detailed probabilities
            st.subheader("📈 Detailed Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.metric(
                    "🐱 Cat Probability",
                    f"{probabilities['Cat']:.2f}%",
                    delta=None
                )
            
            with prob_col2:
                st.metric(
                    "🐶 Dog Probability",
                    f"{probabilities['Dog']:.2f}%",
                    delta=None
                )
        else:
            st.info("👆 Upload an image and click 'Predict' to see results")

# ==================== MULTIPLE IMAGES MODE ====================
def multiple_images_mode(model, conf_threshold):
    """Mode untuk upload multiple gambar"""
    
    st.subheader("📤 Upload Multiple Images")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload beberapa gambar kucing atau anjing"
    )
    
    if uploaded_files:
        st.info(f"📊 Total images uploaded: {len(uploaded_files)}")
        
        if st.button("🔮 Predict All", type="primary"):
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}...")
                
                # Predict
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
            
            # Display results
            st.success(f"✅ Processed {len(results)} images!")
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            cat_count = sum(1 for r in results if r['label'] == 'Cat')
            dog_count = sum(1 for r in results if r['label'] == 'Dog')
            avg_confidence = np.mean([r['confidence'] for r in results])
            high_conf_count = sum(1 for r in results if r['confidence'] >= conf_threshold)
            
            with col1:
                st.metric("🐱 Total Cats", cat_count)
            with col2:
                st.metric("🐶 Total Dogs", dog_count)
            with col3:
                st.metric("📈 Avg Confidence", f"{avg_confidence*100:.1f}%")
            with col4:
                st.metric("✅ High Confidence", f"{high_conf_count}/{len(results)}")
            
            # Display individual results
            st.subheader("🖼️ Individual Results")
            
            # Create grid
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(results):
                        result = results[idx]
                        
                        with cols[j]:
                            st.image(result['image'], use_column_width=True)
                            
                            # Prediction info
                            emoji = "🐱" if result['label'] == "Cat" else "🐶"
                            conf_color = "🟢" if result['confidence'] >= conf_threshold else "🟡"
                            
                            st.markdown(f"""
                            **{result['filename']}**
                            
                            {emoji} **{result['label']}** {conf_color}
                            
                            Confidence: **{result['confidence']*100:.1f}%**
                            """)
            
            # Download results as CSV
            st.subheader("💾 Download Results")
            
            import pandas as pd
            df_results = pd.DataFrame([{
                'Filename': r['filename'],
                'Prediction': r['label'],
                'Confidence (%)': f"{r['confidence']*100:.2f}",
                'Cat Probability (%)': f"{r['probabilities']['Cat']:.2f}",
                'Dog Probability (%)': f"{r['probabilities']['Dog']:.2f}"
            } for r in results])
            
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()