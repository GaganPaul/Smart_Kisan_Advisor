import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import json
import os
from datetime import datetime
import requests
from groq import Groq
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load configuration from Streamlit Secrets
def load_config():
    """Load configuration from Streamlit Secrets"""
    try:
        # Access secrets directly from st.secrets
        config = {
            "api": {
                "groq_api_key": st.secrets.get("GROQ_API_KEY", ""),
                "weather_api_key": st.secrets.get("WEATHER_API_KEY", ""),
                "market_api_key": st.secrets.get("MARKET_API_KEY", "")
            },
            "app": {
                "default_language": st.secrets.get("DEFAULT_LANGUAGE", "English"),
                "model_name": st.secrets.get("MODEL_NAME", "gemma2-9b-it"),
                "max_tokens": st.secrets.get("MAX_TOKENS", 1000),
                "temperature": st.secrets.get("TEMPERATURE", 0.7)
            },
            "features": {
                "enable_disease_detection": st.secrets.get("ENABLE_DISEASE_DETECTION", True),
                "enable_chat_advisor": st.secrets.get("ENABLE_CHAT_ADVISOR", True),
                "enable_weather_info": st.secrets.get("ENABLE_WEATHER_INFO", True),
                "enable_market_prices": st.secrets.get("ENABLE_MARKET_PRICES", True)
            }
        }
        return config
    except Exception as e:
        st.error(f"Error loading configuration from secrets: {e}")
        return None

# Load configuration
config = load_config()

# Page configuration
st.set_page_config(
    page_title="🌾 Smart Kisan Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: black;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: black;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        color: black;
    }
    .prediction-result h4 {
        color: black !important;
    }
    .prediction-result p {
        color: black !important;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: black;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: black;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'disease_results' not in st.session_state:
    st.session_state.disease_results = None
if 'farmer_profile' not in st.session_state:
    st.session_state.farmer_profile = {}
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

# Language translations
TRANSLATIONS = {
    'English': {
        'title': '🌾 Smart Kisan Advisor',
        'subtitle': 'AI-Powered Agricultural Advisory System',
        'disease_detection': 'Crop Disease Detection',
        'chat_advisor': 'AI Chat Advisor',
        'weather_info': 'Weather Information',
        'upload_image': 'Upload Crop Image',
        'analyze': 'Analyze Disease',
        'chat_placeholder': 'Ask me about farming...',
        'send': 'Send',
        'location': 'Location',
        'crop_type': 'Crop Type',
        'soil_type': 'Soil Type',
        'get_advice': 'Get Personalized Advice',
        'weather_forecast': 'Weather Forecast',
        'market_prices': 'Market Prices',
        'government_schemes': 'Government Schemes'
    },
    'Hindi': {
        'title': '🌾 स्मार्ट किसान सलाहकार',
        'subtitle': 'AI-संचालित कृषि सलाहकार प्रणाली',
        'disease_detection': 'फसल रोग पहचान',
        'chat_advisor': 'AI चैट सलाहकार',
        'weather_info': 'मौसम की जानकारी',
        'upload_image': 'फसल की तस्वीर अपलोड करें',
        'analyze': 'रोग का विश्लेषण करें',
        'chat_placeholder': 'खेती के बारे में पूछें...',
        'send': 'भेजें',
        'location': 'स्थान',
        'crop_type': 'फसल का प्रकार',
        'soil_type': 'मिट्टी का प्रकार',
        'get_advice': 'व्यक्तिगत सलाह प्राप्त करें',
        'weather_forecast': 'मौसम पूर्वानुमान',
        'market_prices': 'बाजार मूल्य',
        'government_schemes': 'सरकारी योजनाएं'
    },
    'Punjabi': {
        'title': '🌾 ਸਮਾਰਟ ਕਿਸਾਨ ਸਲਾਹਕਾਰ',
        'subtitle': 'AI-ਚਾਲਿਤ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ ਸਿਸਟਮ',
        'disease_detection': 'ਫਸਲ ਰੋਗ ਪਛਾਣ',
        'chat_advisor': 'AI ਚੈਟ ਸਲਾਹਕਾਰ',
        'weather_info': 'ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ',
        'upload_image': 'ਫਸਲ ਦੀ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ',
        'analyze': 'ਰੋਗ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ',
        'chat_placeholder': 'ਖੇਤੀਬਾੜੀ ਬਾਰੇ ਪੁੱਛੋ...',
        'send': 'ਭੇਜੋ',
        'location': 'ਸਥਾਨ',
        'crop_type': 'ਫਸਲ ਦਾ ਕਿਸਮ',
        'soil_type': 'ਮਿੱਟੀ ਦਾ ਕਿਸਮ',
        'get_advice': 'ਨਿੱਜੀ ਸਲਾਹ ਪ੍ਰਾਪਤ ਕਰੋ',
        'weather_forecast': 'ਮੌਸਮ ਪੂਰਵਾਨੁਮਾਨ',
        'market_prices': 'ਬਾਜ਼ਾਰ ਮੁੱਲ',
        'government_schemes': 'ਸਰਕਾਰੀ ਯੋਜਨਾਵਾਂ'
    }
}

# Initialize Groq client
@st.cache_resource
def initialize_groq():
    try:
        if config is None:
            st.error("Configuration not loaded. Please check Streamlit Secrets.")
            return None
            
        api_key = config.get("api", {}).get("groq_api_key")
        if not api_key:
            st.error("GROQ_API_KEY not found in Streamlit Secrets. Please add it to your secrets configuration.")
            return None
            
        model_name = config.get("app", {}).get("model_name", "gemma2-9b-it")
        
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name
        )
    except Exception as e:
        st.error(f"Error initializing Groq: {e}")
        return None

# Load pre-trained disease detection model
@st.cache_resource
def load_disease_model():
    try:
        # Using a specialized crop disease detection model
        classifier = pipeline(
            "image-classification", 
            model="wambugu71/crop_leaf_diseases_vit",
            top_k=5
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading disease detection model: {e}")
        return None

# Disease detection function
def predict_disease(image, classifier):
    if classifier is None:
        return None, 0.0

    try:
        # Preprocess image
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
            
        if len(img.shape) == 3 and img.shape[-1] == 4:  # RGBA
            img = img[..., :3]
        
        # Convert to PIL Image for the pipeline
        pil_image = Image.fromarray(img)
        
        # Get predictions from the specialized crop disease model
        predictions = classifier(pil_image)
        
        # Extract the top prediction
        if predictions and len(predictions) > 0:
            top_prediction = predictions[0]
            detected_disease = top_prediction['label']
            confidence = top_prediction['score']
            
            # Clean up the disease name for better display
            detected_disease = detected_disease.replace('_', ' ').title()
            
            return detected_disease, confidence
        else:
            return "Unknown Disease", 0.0
        
    except Exception as e:
        st.error(f"Error in disease prediction: {e}")
        return None, 0.0

# Generate agricultural advice using LLM
def generate_advice(disease, crop_type, soil_type, location, llm):
    if llm is None:
        return "LLM service not available. Please check your API key."
    
    try:
        prompt_template = PromptTemplate(
            input_variables=["disease", "crop_type", "soil_type", "location"],
            template="""
            You are an expert agricultural advisor helping Indian farmers. 
            Provide detailed, practical advice in simple language for the following situation:
            
            Disease detected: {disease}
            Crop type: {crop_type}
            Soil type: {soil_type}
            Location: {location}
            
            Please provide:
            1. What this disease means for the crop
            2. Immediate treatment steps (organic and chemical options)
            3. Preventive measures for future
            4. Expected recovery time
            5. When to consult an expert
            
            Keep the advice practical, affordable, and suitable for small farmers in India.
            """
        )
        
        prompt = prompt_template.format(
            disease=disease,
            crop_type=crop_type,
            soil_type=soil_type,
            location=location
        )
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error generating advice: {e}"

# Chat function
def chat_with_advisor(message, llm):
    if llm is None:
        return "LLM service not available. Please check your API key."
    
    try:
        # Add context about being an agricultural advisor
        context = "You are a helpful agricultural advisor for Indian farmers. Provide practical, affordable advice in simple language. Focus on organic solutions when possible and consider the financial constraints of small farmers."
        
        full_prompt = f"{context}\n\nFarmer's question: {message}\n\nProvide a helpful response:"
        
        response = llm.invoke(full_prompt)
        return response.content
        
    except Exception as e:
        return f"Error in chat: {e}"

# Weather information (simulated)
def get_weather_info(location):
    # Simulated weather data - in real implementation, use a weather API
    weather_data = {
        "temperature": "28°C",
        "humidity": "65%",
        "wind_speed": "12 km/h",
        "forecast": "Partly cloudy with light rain expected",
        "recommendations": [
            "Good conditions for spraying pesticides",
            "Avoid irrigation during rain",
            "Monitor for fungal diseases due to humidity"
        ]
    }
    return weather_data

# Market prices (simulated)
def get_market_prices():
    # Simulated market data - in real implementation, use a market API
    prices = {
        "Wheat": "₹2,200/quintal",
        "Rice": "₹1,800/quintal", 
        "Cotton": "₹6,500/quintal",
        "Sugarcane": "₹315/quintal",
        "Potato": "₹1,200/quintal",
        "Tomato": "₹40/kg"
    }
    return prices

# Government schemes (simulated)
def get_government_schemes():
    schemes = [
        {
            "name": "PM-KISAN",
            "description": "Direct income support of ₹6,000 per year to farmers",
            "eligibility": "Small and marginal farmers",
            "application": "Apply through Common Service Centers"
        },
        {
            "name": "PM Fasal Bima Yojana",
            "description": "Crop insurance scheme for farmers",
            "eligibility": "All farmers growing notified crops",
            "application": "Contact nearest bank or insurance company"
        },
        {
            "name": "Soil Health Card Scheme",
            "description": "Free soil testing and recommendations",
            "eligibility": "All farmers",
            "application": "Contact nearest Krishi Vigyan Kendra"
        }
    ]
    return schemes

# Main application
def main():
    # Header
    t = TRANSLATIONS[st.session_state.selected_language]
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for language selection and farmer profile
    with st.sidebar:
        st.header("🌍 Language / भाषा")
        language = st.selectbox(
            "Select Language",
            ["English", "Hindi", "Punjabi"],
            index=["English", "Hindi", "Punjabi"].index(st.session_state.selected_language)
        )
        st.session_state.selected_language = language
        
        st.header("👨‍🌾 Farmer Profile")
        location = st.text_input("Location", value="Punjab, India")
        crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Cotton", "Sugarcane", "Potato", "Tomato", "Other"])
        soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Black", "Red", "Other"])
        
        if st.button("Update Profile"):
            st.session_state.farmer_profile = {
                "location": location,
                "crop_type": crop_type,
                "soil_type": soil_type
            }
            st.success("Profile updated!")
    
    # Initialize models
    llm = initialize_groq()
    disease_classifier = load_disease_model()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 " + t['disease_detection'],
        "💬 " + t['chat_advisor'], 
        "🌤️ " + t['weather_info'],
        "📊 Market & Schemes"
    ])
    
    # Tab 1: Disease Detection
    with tab1:
        st.markdown(f"<div class='feature-card'><h3>{t['disease_detection']}</h3></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                t['upload_image'],
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image of the affected plant part"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button(t['analyze']):
                    with st.spinner("Analyzing image..."):
                        disease, confidence = predict_disease(image, disease_classifier)
                        
                        if disease:
                            st.session_state.disease_results = {
                                "disease": disease,
                                "confidence": confidence,
                                "image": image
                            }
                            st.success("Analysis complete!")
        
        with col2:
            if st.session_state.disease_results:
                result = st.session_state.disease_results
                
                st.markdown(f"""
                <div class='prediction-result'>
                    <h4>🔍 Detection Results</h4>
                    <p><strong>Disease:</strong> {result['disease']}</p>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate advice
                if st.button("Get Treatment Advice"):
                    with st.spinner("Generating personalized advice..."):
                        advice = generate_advice(
                            result['disease'],
                            st.session_state.farmer_profile.get('crop_type', 'Unknown'),
                            st.session_state.farmer_profile.get('soil_type', 'Unknown'),
                            st.session_state.farmer_profile.get('location', 'Unknown'),
                            llm
                        )
                        
                        st.markdown(f"""
                        <div class='feature-card'>
                            <h4>🌱 Treatment & Prevention Advice</h4>
                            <div style='white-space: pre-line;'>{advice}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Tab 2: Chat Advisor
    with tab2:
        st.markdown(f"<div class='feature-card'><h3>{t['chat_advisor']}</h3></div>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message user-message'>
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message ai-message'>
                    <strong>AI Advisor:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            t['chat_placeholder'],
            key="chat_input"
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(t['send']):
                if user_input:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        ai_response = chat_with_advisor(user_input, llm)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    st.rerun()
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Tab 3: Weather Information
    with tab3:
        st.markdown(f"<div class='feature-card'><h3>{t['weather_info']}</h3></div>", unsafe_allow_html=True)
        
        weather_data = get_weather_info(location)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='feature-card'>
                <h4>🌡️ Current Weather</h4>
                <p><strong>Temperature:</strong> """ + weather_data["temperature"] + """</p>
                <p><strong>Humidity:</strong> """ + weather_data["humidity"] + """</p>
                <p><strong>Wind Speed:</strong> """ + weather_data["wind_speed"] + """</p>
                <p><strong>Forecast:</strong> """ + weather_data["forecast"] + """</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='warning-box'>
                <h4>🌾 Farming Recommendations</h4>
            """, unsafe_allow_html=True)
            
            for rec in weather_data["recommendations"]:
                st.markdown(f"• {rec}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 4: Market & Schemes
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<div class='feature-card'><h3>{t['market_prices']}</h3></div>", unsafe_allow_html=True)
            
            prices = get_market_prices()
            for crop, price in prices.items():
                st.markdown(f"**{crop}:** {price}")
        
        with col2:
            st.markdown(f"<div class='feature-card'><h3>{t['government_schemes']}</h3></div>", unsafe_allow_html=True)
            
            schemes = get_government_schemes()
            for scheme in schemes:
                with st.expander(scheme["name"]):
                    st.write(f"**Description:** {scheme['description']}")
                    st.write(f"**Eligibility:** {scheme['eligibility']}")
                    st.write(f"**How to Apply:** {scheme['application']}")

if __name__ == "__main__":
    main()
