# 🌾 Smart Kisan Advisor - Agricultural Advisory System

## Problem Statement
A majority of small and marginal farmers in India rely on traditional knowledge, local shopkeepers, or guesswork for crop selection, pest control, and fertilizer use. They lack access to personalized, real-time advisory services that account for soil type, weather conditions, and crop history.

## Solution Overview
Smart Kisan Advisor is an AI-powered agricultural advisory system that provides:
- **Crop Disease Detection**: Upload images to identify plant diseases
- **AI Chat Advisor**: Get personalized farming advice in multiple languages
- **Weather Information**: Real-time weather data and farming recommendations
- **Market Prices**: Current crop prices and government schemes

## Features

### 🌍 Multilingual Support
- English, Hindi, and Punjabi interfaces
- Localized content and recommendations

### 🔍 Disease Detection
- Upload crop images for instant disease identification
- Uses pre-trained models from Hugging Face
- Provides confidence scores and treatment recommendations

### 💬 AI Chat Advisor
- Powered by Groq LLM for intelligent responses
- Context-aware conversations
- Practical, affordable advice for small farmers

### 🌤️ Weather Integration
- Current weather conditions
- Farming-specific recommendations
- Seasonal alerts and predictions

### 📊 Market & Government Information
- Real-time crop prices
- Government schemes and eligibility
- Application procedures

## Technology Stack

### Core Framework
- **Streamlit**: Web application framework
- **Python 3.12**: Backend programming language

### AI/ML Components
- **Groq LLM**: Fast inference for chat functionality
- **LangChain**: Prompt management and conversation memory
- **Transformers**: Pre-trained models for disease detection
- **TensorFlow**: Deep learning framework

### UI/UX
- **Streamlit Components**: Modern, responsive interface
- **Custom CSS**: Beautiful gradients and animations
- **Multilingual Support**: Seamless language switching

## Installation & Setup

### Prerequisites
- Python 3.12 or higher
- Groq API key (get from https://console.groq.com/)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd infogri
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Environment Setup
Create a `.env` file in the project root:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here
```

### Step 4: Run the Application
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501`

## Usage Guide

### 1. Disease Detection
1. Navigate to the "Crop Disease Detection" tab
2. Upload a clear image of the affected plant part
3. Click "Analyze Disease" to get results
4. View detection results and confidence scores
5. Get personalized treatment advice

### 2. AI Chat Advisor
1. Go to the "AI Chat Advisor" tab
2. Type your farming-related question
3. Receive instant, contextual advice
4. Continue the conversation for detailed guidance

### 3. Weather Information
1. Check the "Weather Information" tab
2. View current weather conditions
3. Read farming-specific recommendations
4. Plan activities based on weather forecasts

### 4. Market & Schemes
1. Visit the "Market & Schemes" tab
2. Check current crop prices
3. Browse government schemes
4. Learn about eligibility and application procedures

## Project Structure
```
infogri/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env.example           # Environment variables template
├── data/                  # Data storage (future use)
├── models/                # ML models (future use)
└── .gitignore            # Git ignore file
```

## Future Enhancements

### Phase 2: Mobile App (Flutter)
- Cross-platform mobile application
- Offline disease detection
- Push notifications for weather alerts
- Voice input/output for low-literacy users

### Phase 3: Advanced Features
- RAG system for government schemes
- Real-time weather API integration
- Market price APIs
- Soil health recommendations
- Crop yield predictions

### Phase 4: Enterprise Features
- Multi-tenant architecture
- Admin dashboard
- Analytics and reporting
- Integration with government databases

## Revenue Model
- **Free for Farmers**: No cost to small and marginal farmers
- **Government Partnerships**: Revenue through government contracts
- **Corporate Partnerships**: Premium features for agri-businesses
- **Data Analytics**: Anonymized insights for research institutions

## Contributing
This project is developed for SIH 2025 Hackathon. Contributions are welcome!

## License
This project is licensed under the MIT License.

## Contact
For questions and support, please reach out to the development team.

---

**Built with ❤️ for Indian Farmers**
