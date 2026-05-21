import streamlit as st
import os
import tempfile
import time
import numpy as np
from audio_recorder_streamlit import audio_recorder
import librosa
import soundfile as sf

# Set page configuration
st.set_page_config(
    page_title="Deep Fake Voice Detection System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Main styling */
.stApp {
    background: linear-gradient(135deg, #0F0C29 0%, #24243e 50%, #302B63 100%);
    font-family: 'Inter', sans-serif;
    color: #FFFFFF;
    min-height: 100vh;
}

/* Remove default streamlit padding */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Header styling */
.main-header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 3rem 0;
    position: relative;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 2px;
}

.main-header h1 {
    color: #FFFFFF;
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
    text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

.main-header p {
    color: rgba(255,255,255,0.8);
    font-size: 1.3rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 8px;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 15px;
    color: rgba(255, 255, 255, 0.7);
    padding: 12px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    border: none;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #FFFFFF;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

/* Module card styling */
.module-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(30px);
    border-radius: 30px;
    padding: 2.5rem;
    box-shadow: 
        0 25px 50px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.module-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
}

.module-card:hover {
    transform: translateY(-8px);
    box-shadow: 
        0 35px 70px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.15);
    border-color: rgba(255,255,255,0.15);
}

/* Module header */
.module-header {
    display: flex;
    align-items: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.module-icon {
    width: 60px;
    height: 60px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 20px;
    font-size: 28px;
    color: white;
    position: relative;
    overflow: hidden;
}

.module-icon::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: linear-gradient(45deg, rgba(255,255,255,0.2), transparent);
    border-radius: 20px;
    z-index: -1;
}

.icon-detection { 
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
    box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4);
}

.icon-matcher { 
    background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.4);
}

.icon-urdu { 
    background: linear-gradient(135deg, #FECA57 0%, #FF9FF3 100%);
    box-shadow: 0 10px 25px rgba(254, 202, 87, 0.4);
}

.module-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}

.module-description {
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
    font-weight: 400;
}

/* Section titles */
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* File uploader styling */
.stFileUploader > div {
    background: rgba(255, 255, 255, 0.05);
    border: 2px dashed rgba(102, 126, 234, 0.5);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.1);
    transform: translateY(-2px);
}

.stFileUploader label {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 500;
}

/* Button styling */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 18px 32px;
    border-radius: 18px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 1.5rem;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.6s;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:active {
    transform: translateY(-1px);
}

/* Audio player styling */
.stAudio {
    margin: 1rem 0;
}

/* File info styling */
.file-info {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.file-name {
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 8px;
    font-size: 1rem;
}

.file-details {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
}

/* Results styling */
.result-section {
    margin-top: 2rem;
    padding: 2rem;
    border-radius: 20px;
    border-left: 4px solid;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.result-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    opacity: 0.1;
    background-size: 20px 20px;
    background-image: 
        linear-gradient(45deg, rgba(255,255,255,0.1) 25%, transparent 25%),
        linear-gradient(-45deg, rgba(255,255,255,0.1) 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.1) 75%),
        linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.1) 75%);
    z-index: -1;
}

.result-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.result-content {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
    line-height: 1.8;
}

.result-content strong {
    font-weight: 700;
    text-shadow: 0 0 10px currentColor;
}

.result-fake {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 107, 107, 0.05) 100%);
    box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
}

.result-real {
    border-left-color: #4ECDC4;
    background: linear-gradient(135deg, rgba(78, 205, 196, 0.15) 0%, rgba(78, 205, 196, 0.05) 100%);
    box-shadow: 0 10px 30px rgba(78, 205, 196, 0.2);
}

.result-match {
    border-left-color: #4ECDC4;
    background: linear-gradient(135deg, rgba(78, 205, 196, 0.15) 0%, rgba(78, 205, 196, 0.05) 100%);
    box-shadow: 0 10px 30px rgba(78, 205, 196, 0.2);
}

.result-nomatch {
    border-left-color: #FF6B6B;
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 107, 107, 0.05) 100%);
    box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
}

/* Column styling */
.stColumn {
    padding: 0 1rem;
}

/* Spinner styling */
.stSpinner > div {
    border-color: rgba(102, 126, 234, 0.2) !important;
    border-top-color: #667eea !important;
}

/* Warning styling */
.stAlert {
    background: rgba(254, 202, 87, 0.1);
    border: 1px solid rgba(254, 202, 87, 0.3);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.stAlert [data-testid="alert-content"] {
    color: #FECA57;
}

/* Matcher specific styling */
.matcher-label {
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Audio recorder custom styling */
.stAudioRecorder {
    display: flex;
    justify-content: center;
    margin: 1.5rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem;
    }
    
    .module-card {
        padding: 1.5rem;
        margin: 0 0.5rem 1.5rem;
    }
    
    .module-icon {
        width: 50px;
        height: 50px;
        font-size: 24px;
    }
    
    .module-title {
        font-size: 1.3rem;
    }
}

@media (max-width: 480px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header p {
        font-size: 1rem;
    }
    
    .module-card {
        padding: 1rem;
        margin: 0 0.25rem 1rem;
    }
}

/* Glass morphism effects */
.glass-effect {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
}

/* Animated gradient backgrounds */
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.animated-bg {
    background: linear-gradient(-45deg, #667eea, #764ba2, #667eea, #764ba2);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

/* Floating particles effect */
.particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: -1;
}

.particle {
    position: absolute;
    background: rgba(102, 126, 234, 0.3);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

/* Glow effects */
.glow-text {
    text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
}

.glow-border {
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
}

/* Success/Error indicators */
.success-indicator {
    color: #4ECDC4;
    text-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
}

.error-indicator {
    color: #FF6B6B;
    text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Enhanced file upload area */
.upload-enhanced {
    background: rgba(255, 255, 255, 0.03);
    border: 2px dashed rgba(102, 126, 234, 0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}

.upload-enhanced::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
    transform: rotate(45deg);
    transition: all 0.6s ease;
    opacity: 0;
}

.upload-enhanced:hover::before {
    opacity: 1;
    transform: rotate(45deg) translateX(100%);
}

.upload-enhanced:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.1);
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
}

/* Text effects */
.gradient-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.neon-text {
    color: #FFFFFF;
    text-shadow: 
        0 0 5px rgba(102, 126, 234, 0.8),
        0 0 10px rgba(102, 126, 234, 0.6),
        0 0 15px rgba(102, 126, 234, 0.4);
}

/* Enhanced animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

/* Custom tab panels */
.stTabs [data-baseweb="tab-panel"] {
    padding: 0;
    background: transparent;
}

/* Sidebar styling if expanded */
.css-1d391kg {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
}

/* Input field styling */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    color: #FFFFFF;
    padding: 12px 16px;
}

.stTextInput > div > div > input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
}

/* Select box styling */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
}

/* Progress bar styling */
.stProgress .css-1aumxhk {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

/* Custom checkbox/radio styling */
.stCheckbox, .stRadio {
    color: #FFFFFF;
}

/* Enhanced spacing */
.stColumn > div {
    padding: 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Import your modules (these would be your actual detection functions)
def deepfake_detect(audio_path):
    """Simulate deepfake detection"""
    time.sleep(2)  # Simulate processing time
    is_fake = np.random.random() > 0.5
    confidence = (np.random.random() * 0.3 + 0.7) * 100
    return is_fake, confidence

def voice_match(audio_path1, audio_path2):
    """Simulate voice matching"""
    time.sleep(2)  # Simulate processing time
    is_match = np.random.random() > 0.4
    similarity = (np.random.random() * 0.4 + 0.6) * 100
    return is_match, similarity

def urdu_detect(audio_path):
    """Simulate Urdu deepfake detection"""
    time.sleep(2)  # Simulate processing time
    is_fake = np.random.random() > 0.6
    confidence = (np.random.random() * 0.25 + 0.75) * 100
    return is_fake, confidence

# Function to save uploaded file
def save_uploaded_file(uploaded_file, folder="temp"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to save recorded audio
def save_recorded_audio(audio_bytes, filename="recorded_audio.wav", folder="temp"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, filename)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header fade-in-up">
        <h1 class="neon-text">🎵 Deep Fake Voice Detection System</h1>
        <p>Advanced AI-powered voice authentication and analysis platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different modules
    tab1, tab2, tab3 = st.tabs([
        "🔍 Voice Authenticity Detection", 
        "🔄 Voice Matching System", 
        "🕌 Urdu Voice Detection"
    ])
    
    with tab1:
        st.markdown("""
        <div class="module-card fade-in-up">
            <div class="module-header">
                <div class="module-icon icon-detection">🔍</div>
                <div>
                    <div class="module-title">Voice Authenticity Detection</div>
                    <div class="module-description">Classify voices as Real or Fake using advanced AI</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-title">📁 Upload Audio File</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"], key="file1")
            
            if uploaded_file is not None:
                file_path = save_uploaded_file(uploaded_file)
                st.audio(uploaded_file, format='audio/wav')
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">📄 {uploaded_file.name}</div>
                    <div class="file-details">📊 Size: {uploaded_file.size / 1024:.2f} KB | 🎵 Type: {uploaded_file.type}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-title">🎙️ Record Audio</div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(
                text="Click to start recording",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                key="recorder1"
            )
            
            if audio_bytes:
                recorded_file = save_recorded_audio(audio_bytes, "recorded1.wav")
                st.audio(audio_bytes, format="audio/wav")
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">🎤 recorded_audio.wav</div>
                    <div class="file-details">✅ Recorded audio ready for analysis</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze Voice", key="analyze1", use_container_width=True):
            if uploaded_file is not None or audio_bytes is not None:
                with st.spinner("Analyzing voice authenticity..."):
                    # Use the uploaded file if available, otherwise use recorded audio
                    audio_path = save_uploaded_file(uploaded_file) if uploaded_file is not None else recorded_file
                    is_fake, confidence = deepfake_detect(audio_path)
                    
                    if is_fake:
                        st.markdown(f"""
                        <div class="result-section result-fake">
                            <div class="result-title">🚨 Analysis Result</div>
                            <div class="result-content">
                                🎭 Voice Classification: <strong class="error-indicator">FAKE</strong><br>
                                📊 Confidence Score: <strong>{confidence:.1f}%</strong><br>
                                🔬 Analysis: Deepfake indicators detected in voice patterns
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-section result-real">
                            <div class="result-title">✅ Analysis Result</div>
                            <div class="result-content">
                                🎭 Voice Classification: <strong class="success-indicator">REAL</strong><br>
                                📊 Confidence Score: <strong>{confidence:.1f}%</strong><br>
                                🔬 Analysis: Voice appears to be authentic
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please upload an audio file or record audio first.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="module-card fade-in-up">
            <div class="module-header">
                <div class="module-icon icon-matcher">🔄</div>
                <div>
                    <div class="module-title">Voice Matching System</div>
                    <div class="module-description">Compare two voices to determine if they match</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="matcher-label">🎯 Voice Sample 1</div>', unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Choose first audio file", type=["wav", "mp3", "m4a"], key="file2a")
            
            if uploaded_file1 is not None:
                file_path1 = save_uploaded_file(uploaded_file1)
                st.audio(uploaded_file1, format='audio/wav')
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">📄 {uploaded_file1.name}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">🎙️ Record Audio</div>', unsafe_allow_html=True)
            audio_bytes1 = audio_recorder(
                text="Record First Voice",
                recording_color="#4ecdc4",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                key="recorder2a"
            )
            
            if audio_bytes1:
                recorded_file1 = save_recorded_audio(audio_bytes1, "recorded2a.wav")
                st.audio(audio_bytes1, format="audio/wav")
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">🎤 recorded_audio1.wav</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="matcher-label">🎯 Voice Sample 2</div>', unsafe_allow_html=True)
            uploaded_file2 = st.file_uploader("Choose second audio file", type=["wav", "mp3", "m4a"], key="file2b")
            
            if uploaded_file2 is not None:
                file_path2 = save_uploaded_file(uploaded_file2)
                st.audio(uploaded_file2, format='audio/wav')
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">📄 {uploaded_file2.name}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">🎙️ Record Audio</div>', unsafe_allow_html=True)
            audio_bytes2 = audio_recorder(
                text="Record Second Voice",
                recording_color="#4ecdc4",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                key="recorder2b"
            )
            
            if audio_bytes2:
                recorded_file2 = save_recorded_audio(audio_bytes2, "recorded2b.wav")
                st.audio(audio_bytes2, format="audio/wav")
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">🎤 recorded_audio2.wav</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔄 Compare Voices", key="analyze2", use_container_width=True):
            # Determine which files to use (uploaded or recorded)
            has_file1 = uploaded_file1 is not None or audio_bytes1 is not None
            has_file2 = uploaded_file2 is not None or audio_bytes2 is not None
            
            if has_file1 and has_file2:
                with st.spinner("Comparing voice samples..."):
                    # Get file paths
                    audio_path1 = save_uploaded_file(uploaded_file1) if uploaded_file1 is not None else recorded_file1
                    audio_path2 = save_uploaded_file(uploaded_file2) if uploaded_file2 is not None else recorded_file2
                    
                    is_match, similarity = voice_match(audio_path1, audio_path2)
                    
                    if is_match:
                        st.markdown(f"""
                        <div class="result-section result-match">
                            <div class="result-title">✅ Comparison Result</div>
                            <div class="result-content">
                                🎭 Voice Comparison: <strong class="success-indicator">MATCH</strong><br>
                                📊 Similarity Score: <strong>{similarity:.1f}%</strong><br>
                                🔬 Analysis: Voice samples appear to be from the same speaker
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-section result-nomatch">
                            <div class="result-title">❌ Comparison Result</div>
                            <div class="result-content">
                                🎭 Voice Comparison: <strong class="error-indicator">NO MATCH</strong><br>
                                📊 Similarity Score: <strong>{similarity:.1f}%</strong><br>
                                🔬 Analysis: Voice samples are from different speakers
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please provide two voice samples for comparison.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="module-card fade-in-up">
            <div class="module-header">
                <div class="module-icon icon-urdu">🕌</div>
                <div>
                    <div class="module-title">Urdu Voice Detection</div>
                    <div class="module-description">Specialized deepfake detection for Urdu language</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-title">📁 Upload Urdu Audio</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an Urdu audio file", type=["wav", "mp3", "m4a"], key="file3")
            
            if uploaded_file is not None:
                file_path = save_uploaded_file(uploaded_file)
                st.audio(uploaded_file, format='audio/wav')
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">📄 {uploaded_file.name}</div>
                    <div class="file-details">📊 Size: {uploaded_file.size / 1024:.2f} KB | 🎵 Type: {uploaded_file.type}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="section-title">🎙️ Record Urdu Speech</div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(
                text="Click to record Urdu speech",
                recording_color="#feca57",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                key="recorder3"
            )
            
            if audio_bytes:
                recorded_file = save_recorded_audio(audio_bytes, "recorded3.wav")
                st.audio(audio_bytes, format="audio/wav")
                st.markdown(f"""
                <div class="file-info">
                    <div class="file-name">🎤 recorded_urdu.wav</div>
                    <div class="file-details">✅ Recorded Urdu audio ready for analysis</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze Urdu Voice", key="analyze3", use_container_width=True):
            has_audio = uploaded_file is not None or audio_bytes is not None
            
            if has_audio:
                with st.spinner("Analyzing Urdu voice authenticity..."):
                    # Use the uploaded file if available, otherwise use recorded audio
                    audio_path = save_uploaded_file(uploaded_file) if uploaded_file is not None else recorded_file
                    is_fake, confidence = urdu_detect(audio_path)
                    
                    if is_fake:
                        st.markdown(f"""
                        <div class="result-section result-fake">
                            <div class="result-title">🚨 Analysis Result</div>
                            <div class="result-content">
                                🎭 Urdu Voice Classification: <strong class="error-indicator">FAKE</strong><br>
                                📊 Confidence Score: <strong>{confidence:.1f}%</strong><br>
                                🌐 Language: Urdu<br>
                                🔬 Analysis: Deepfake patterns detected in Urdu speech
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-section result-real">
                            <div class="result-title">✅ Analysis Result</div>
                            <div class="result-content">
                                🎭 Urdu Voice Classification: <strong class="success-indicator">REAL</strong><br>
                                📊 Confidence Score: <strong>{confidence:.1f}%</strong><br>
                                🌐 Language: Urdu<br>
                                🔬 Analysis: Authentic Urdu voice detected
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please upload an Urdu audio file or record Urdu speech first.")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()