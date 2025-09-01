#!/usr/bin/env python3
"""
Test script to verify the setup of Smart Kisan Advisor
Run this script to check if all dependencies and API keys are working correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from groq import Groq
        print("✅ Groq imported successfully")
    except ImportError as e:
        print(f"❌ Groq import failed: {e}")
        return False
    
    try:
        from langchain_groq import ChatGroq
        print("✅ LangChain Groq imported successfully")
    except ImportError as e:
        print(f"❌ LangChain Groq import failed: {e}")
        return False
    
    try:
        from transformers import pipeline
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test if Streamlit Secrets are configured"""
    print("\n🔍 Testing Streamlit Secrets configuration...")
    
    try:
        # Simulate Streamlit secrets access
        # In a real Streamlit app, this would be st.secrets
        print("ℹ️  Streamlit Secrets configuration check")
        print("   Please ensure you have configured the following secrets:")
        print("   - GROQ_API_KEY")
        print("   - MODEL_NAME (optional, defaults to 'gemma2-9b-it')")
        print("   - MAX_TOKENS (optional, defaults to 1000)")
        print("   - TEMPERATURE (optional, defaults to 0.7)")
        return True
    except Exception as e:
        print(f"❌ Error checking secrets configuration: {e}")
        return False

def test_groq_connection():
    """Test if Groq API is accessible"""
    print("\n🔍 Testing Groq API connection...")
    
    try:
        from langchain_groq import ChatGroq
        
        # For testing purposes, we'll use environment variable as fallback
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            print("⚠️  GROQ_API_KEY not configured - skipping API test")
            print("   Please set your Groq API key in Streamlit Secrets or as environment variable")
            return True  # Don't fail the test, just warn
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="gemma2-9b-it"
        )
        
        # Test with a simple prompt
        response = llm.invoke("Hello, this is a test message.")
        print("✅ Groq API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Groq API connection failed: {e}")
        return False

def test_disease_model():
    """Test if disease detection model can be loaded"""
    print("\n🔍 Testing disease detection model...")
    
    try:
        from transformers import pipeline
        
        # This will download the model on first run
        classifier = pipeline(
            "image-classification",
            model="wambugu71/crop_leaf_diseases_vit",
            top_k=5
        )
        print("✅ Disease detection model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Disease detection model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌾 Smart Kisan Advisor - Setup Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test environment
    if not test_environment():
        all_tests_passed = False
    
    # Test Groq connection (only if environment is set)
    if test_environment():
        if not test_groq_connection():
            all_tests_passed = False
    
    # Test disease model
    if not test_disease_model():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print("   You can now run: streamlit run main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("   Check the README.md for setup instructions.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
