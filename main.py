import streamlit as st
import json
import os
import requests
from datetime import datetime
import re
from transformers import pipeline
import torch
import google.generativeai as genai

# Configure Gemini from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, AttributeError):
    st.error("Gemini API key not found. Please set it in Streamlit secrets.toml file.")
    st.stop()

class MedicalChatbot:
    def __init__(self):
        self.sentiment_analyzer = self.load_sentiment_analyzer()
        self.users_file = "kb/users.json"
        self.diet_file = "diet.json"
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # PROMPT 1: System Role
        self.system_prompt = """
        You are a medical health information assistant.
        You are NOT a doctor.
        You must NOT diagnose diseases or prescribe medication.
        You provide general, educational health information only.

        Rules:
        - Always include a medical disclaimer
        - Encourage consulting licensed healthcare professionals
        - Use cautious, non-definitive language
        - Treat mental health and emergency symptoms with extra care
        - Advise immediate medical attention for severe or alarming symptoms

        Tone:
        Calm, professional, empathetic.
        """
        
        # PROMPT 8: UI Disclaimer (for display)
        self.ui_disclaimer = """
        ⚠️ This chatbot provides general health information only.
        It does not provide medical diagnoses or treatment.
        Always consult a licensed healthcare professional for medical concerns.
        """
        
        self.load_data()
        
    def load_sentiment_analyzer(self):
        """Load sentiment analyzer with error handling"""
        try:
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        except:
            return None
        
    def load_data(self):
        """Load users and diet data"""
        # Load users
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        except:
            self.users = {}
            
        # Load diet data
        try:
            with open(self.diet_file, 'r') as f:
                self.diet_data = json.load(f)
        except:
            self.diet_data = {}
    
    def save_users(self):
        """Save users data to file"""
        os.makedirs('kb', exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if self.sentiment_analyzer and text:
            try:
                result = self.sentiment_analyzer(text[:512])
                return result[0]
            except:
                return {'label': 'NEUTRAL', 'score': 0.5}
        return {'label': 'NEUTRAL', 'score': 0.5}
    
    def register_user(self, user_data):
        """Register a new user"""
        phone = user_data['mobile_number']
        if phone in self.users:
            return False, "User already exists with this phone number"
        
        # PROMPT 9: Fix field naming - store age properly
        self.users[phone] = {
            'Name': user_data['name'],
            'Age': user_data['age'],  # Changed from 'DOB' to 'Age'
            'Height': user_data['height'],
            'Weight': user_data['weight'],
            'Gender': user_data['gender'],
            'Country': user_data['country'],
            'MedicalInfo': "",
            'RegistrationDate': datetime.now().isoformat(),
            'DietHistory': []
        }
        self.save_users()
        return True, "Registration successful!"
    
    def login_user(self, phone_number):
        """Login existing user"""
        if phone_number in self.users:
            return True, self.users[phone_number]
        return False, "User not found"
    
    def update_medical_info(self, phone_number, medical_text):
        """Update user's medical information"""
        if phone_number in self.users:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            if self.users[phone_number]['MedicalInfo']:
                self.users[phone_number]['MedicalInfo'] += f"\n{timestamp}: {medical_text}"
            else:
                self.users[phone_number]['MedicalInfo'] = f"{timestamp}: {medical_text}"
            self.save_users()
            return True
        return False
    
    def update_diet_history(self, phone_number, diet_info):
        """Update user's diet history"""
        if phone_number in self.users:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            diet_entry = {
                'timestamp': timestamp,
                'diet_info': diet_info,
                'analysis': self.analyze_diet_pattern(diet_info)
            }
            self.users[phone_number]['DietHistory'].append(diet_entry)
            self.save_users()
    
    def analyze_diet_pattern(self, diet_text):
        """Analyze diet pattern for unhealthy foods"""
        diet_lower = diet_text.lower()
        unhealthy_found = []
        recommendations = []
        
        for food in self.diet_data.get('unhealthy_foods', []):
            if food in diet_lower:
                unhealthy_found.append(food)
                if food in self.diet_data.get('healthy_alternatives', {}):
                    recommendations.extend(self.diet_data['healthy_alternatives'][food])
        
        return {
            'unhealthy_foods': unhealthy_found,
            'recommendations': list(set(recommendations))[:3]  # Limit to 3 recommendations
        }
    
    def detect_intent(self, message):
        """Detect user intent from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['diet', 'nutrition', 'food', 'eat', 'meal', 'calorie', 'sweet', 'fries']):
            return 'nutrition'
        elif any(word in message_lower for word in ['symptom', 'pain', 'hurt', 'fever', 'headache', 'sick', 'ache', 'nausea', 'vomit', 'dizziness']):
            return 'symptom'
        elif any(word in message_lower for word in ['stress', 'anxiety', 'mood', 'feel', 'emotional', 'depress']):
            return 'mental_health'
        elif any(word in message_lower for word in ['weight', 'bmi', 'exercise', 'fitness', 'workout']):
            return 'fitness'
        else:
            return 'general'
    
    # PROMPT 2: Single Entry Point for all Gemini calls
    def _call_gemini(self, task_prompt, user_data=None):
        """Single entry point for all Gemini API calls"""
        try:
            # Prepare context with system prompt and task prompt
            full_prompt = f"{self.system_prompt}\n\n{task_prompt}"
            
            response = self.gemini_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            # Fallback responses based on intent
            if "nutrition" in task_prompt.lower():
                return "I apologize, but I'm having trouble accessing nutrition advice right now. Please try again later or consult a nutritionist."
            elif "symptom" in task_prompt.lower():
                return "I'm currently unable to analyze symptoms. Please consult a healthcare professional for any medical concerns."
            else:
                return "I'm having trouble processing your request. Please try again or consult a healthcare professional for medical advice."
    
    def get_nutrition_advice(self, user_data, diet_context, user_query):
        """Get personalized nutrition advice using Gemini API"""
        # PROMPT 3: Nutrition Task Prompt
        task_prompt = f"""
        Using the user profile and diet details provided:
        - Identify unhealthy dietary patterns
        - Suggest healthier alternatives
        - Recommend balanced meals
        - Personalize advice based on age, weight, height, gender, and country

        User Profile:
        - Name: {user_data['Name']}
        - Age: {user_data['Age']}
        - Height: {user_data['Height']} cm
        - Weight: {user_data['Weight']} kg
        - Gender: {user_data['Gender']}
        - Country: {user_data['Country']}
        
        Current Diet: {diet_context}
        User Query: {user_query}

        Do not diagnose or prescribe.
        Include a medical disclaimer.
        """
        
        return self._call_gemini(task_prompt, user_data)
    
    def get_symptom_analysis(self, symptoms_text, user_data=None):
        """Analyze symptoms using Gemini API"""
        # PROMPT 4: Symptom Analysis Task Prompt
        user_context = ""
        if user_data:
            user_context = f"""
            User Profile:
            - Age: {user_data['Age']}
            - Gender: {user_data['Gender']}
            - Height: {user_data['Height']} cm
            - Weight: {user_data['Weight']} kg
            """
        
        task_prompt = f"""
        The user is describing health symptoms.
        
        {user_context}
        Symptoms described: {symptoms_text}

        Provide:
        1. Possible common causes (not a diagnosis)
        2. General self-care guidance
        3. Warning signs that require medical attention
        4. Important precautions

        Always include a disclaimer stating this is not medical advice.
        Use empathetic and cautious language.
        """
        
        return self._call_gemini(task_prompt, user_data)
    
    def get_mental_health_support(self, user_input, user_data=None):
        """Get mental health support using Gemini API"""
        # PROMPT 5: Mental Health Support Prompt
        user_context = ""
        if user_data:
            user_context = f"""
            User Profile:
            - Age: {user_data['Age']}
            - Gender: {user_data['Gender']}
            """
        
        task_prompt = f"""
        Provide supportive, non-judgmental guidance for mental health concerns.
        
        {user_context}
        User's concern: {user_input}

        Focus on:
        - Coping strategies
        - Breathing or grounding techniques
        - Sleep and routine
        - When to seek professional help

        Do not provide therapy or diagnosis.
        Include a mental health safety disclaimer.
        """
        
        return self._call_gemini(task_prompt, user_data)
    
    def get_fitness_guidance(self, user_input, user_data):
        """Get fitness guidance using Gemini API"""
        # PROMPT 6: Fitness Guidance Prompt
        task_prompt = f"""
        Provide general fitness guidance based on the user profile.
        
        User Profile:
        - Age: {user_data['Age']}
        - Gender: {user_data['Gender']}
        - Height: {user_data['Height']} cm
        - Weight: {user_data['Weight']} kg
        
        Query: {user_input}

        Include:
        - Safe exercise suggestions
        - Activity frequency
        - Precautions based on age and weight

        Do not create medical or rehabilitation plans.
        Include a disclaimer.
        """
        
        return self._call_gemini(task_prompt, user_data)
    
    def get_general_health_info(self, user_input, user_data=None):
        """Get general health information using Gemini API"""
        # PROMPT 7: General Health Prompt
        user_context = ""
        if user_data:
            user_context = f"""
            User Profile (if relevant):
            - Age: {user_data['Age']}
            - Gender: {user_data['Gender']}
            """
        
        task_prompt = f"""
        Provide general, practical health information related to the user's question.
        
        {user_context}
        User's question: {user_input}

        If the input is a greeting, respond politely.
        Avoid diagnosis, certainty, or medical prescriptions.
        Include a disclaimer when health advice is given.
        """
        
        return self._call_gemini(task_prompt, user_data)

    def generate_response(self, user_input, user_data=None, intent=None):
        """Generate personalized response using Gemini API"""
        if not intent:
            intent = self.detect_intent(user_input)
        
        sentiment = self.analyze_sentiment(user_input)
        
        # Empathetic opening based on sentiment
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.7:
            empathetic_opening = "I understand this might be concerning. "
        elif sentiment['label'] == 'NEGATIVE':
            empathetic_opening = "I'm sorry to hear you're feeling this way. "
        else:
            empathetic_opening = "Thank you for sharing. "
        
        # Intent-based responses using single entry point
        if intent == 'nutrition' and user_data:
            # Update diet history
            self.update_diet_history(st.session_state.current_user, user_input)
            
            # Get Gemini advice
            gemini_response = self.get_nutrition_advice(user_data, user_input, user_input)
            
            # Add personalized BMI info
            try:
                height_m = float(user_data['Height']) / 100
                weight_kg = float(user_data['Weight'])
                bmi = weight_kg / (height_m ** 2)
                bmi_info = f"\n\nYour BMI: {bmi:.1f} - "
                if bmi < 18.5:
                    bmi_info += "Consider increasing nutrient-dense foods."
                elif bmi <= 25:
                    bmi_info += "Great! Maintain your healthy weight."
                else:
                    bmi_info += "Consider portion control and increased activity."
            except:
                bmi_info = ""
            
            return empathetic_opening + gemini_response + bmi_info
        
        elif intent == 'symptom':
            # Get Gemini symptom analysis
            gemini_response = self.get_symptom_analysis(user_input, user_data)
            return empathetic_opening + gemini_response
        
        elif intent == 'mental_health':
            # Get mental health support
            gemini_response = self.get_mental_health_support(user_input, user_data)
            return empathetic_opening + gemini_response
        
        elif intent == 'fitness' and user_data:
            # Get fitness guidance
            gemini_response = self.get_fitness_guidance(user_input, user_data)
            return empathetic_opening + gemini_response
        
        else:
            # General health advice
            gemini_response = self.get_general_health_info(user_input, user_data)
            return empathetic_opening + gemini_response

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical Health Chatbot",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'medical_report_mode' not in st.session_state:
        st.session_state.medical_report_mode = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    st.sidebar.title("🏥 Medical Chatbot")
    
    # PROMPT 8: Display UI Disclaimer in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚠️ Important Disclaimer", expanded=True):
        st.markdown(st.session_state.chatbot.ui_disclaimer)
    
    # Login/Register section
    if st.session_state.current_user is None:
        st.sidebar.subheader("Authentication")
        auth_option = st.sidebar.radio("Choose option:", ["Login", "Register"])
        
        if auth_option == "Login":
            phone = st.sidebar.text_input("Mobile Number", placeholder="Enter your registered phone number")
            if st.sidebar.button("Login"):
                if phone:
                    success, user_data = st.session_state.chatbot.login_user(phone)
                    if success:
                        st.session_state.current_user = phone
                        st.session_state.chat_history = []
                        st.sidebar.success("Login successful!")
                        st.rerun()
                    else:
                        st.sidebar.error("User not found")
                else:
                    st.sidebar.warning("Please enter phone number")
        
        else:  # Register
            with st.sidebar.form("register_form"):
                st.subheader("Register New User")
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=1, max_value=120)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                country = st.text_input("Country")
                height = st.number_input("Height (cm)", min_value=50, max_value=250)
                weight = st.number_input("Weight (kg)", min_value=10, max_value=200)
                mobile = st.text_input("Mobile Number")
                
                if st.form_submit_button("Register"):
                    if all([name, age, gender, country, height, weight, mobile]):
                        user_data = {
                            'name': name,
                            'age': str(age),
                            'gender': gender,
                            'country': country,
                            'height': str(height),
                            'weight': str(weight),
                            'mobile_number': mobile
                        }
                        success, message = st.session_state.chatbot.register_user(user_data)
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(message)
                    else:
                        st.error("Please fill all fields")
    
    else:  # User is logged in
        st.sidebar.success(f"Logged in as: {st.session_state.current_user}")
        user_data = st.session_state.chatbot.users[st.session_state.current_user]
        
        # User info display - Updated field name from DOB to Age
        st.sidebar.subheader("User Information")
        st.sidebar.write(f"*Name:* {user_data['Name']}")
        st.sidebar.write(f"*Age:* {user_data['Age']}")
        st.sidebar.write(f"*Height:* {user_data['Height']} cm")
        st.sidebar.write(f"*Weight:* {user_data['Weight']} kg")
        st.sidebar.write(f"*Gender:* {user_data['Gender']}")
        st.sidebar.write(f"*Country:* {user_data['Country']}")
        
        # Medical report mode
        if st.sidebar.button("📝 Medical Report Mode" if not st.session_state.medical_report_mode else "⏹ Stop Medical Report"):
            st.session_state.medical_report_mode = not st.session_state.medical_report_mode
            st.rerun()
        
        # Diet history
        if user_data.get('DietHistory'):
            st.sidebar.subheader("Recent Diet Entries")
            for entry in user_data['DietHistory'][-3:]:  # Last 3 entries
                st.sidebar.write(f"{entry['timestamp']}:** {entry['diet_info'][:50]}...")
        
        # Logout
        if st.sidebar.button("🚪 Logout"):
            st.session_state.current_user = None
            st.session_state.medical_report_mode = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat area
    st.title("💬 Medical Health Assistant")
    
    # PROMPT 8: Display UI disclaimer in main area
    st.markdown("---")
    with st.expander("⚠️ Important Medical Disclaimer", expanded=False):
        st.markdown(st.session_state.chatbot.ui_disclaimer)
    st.markdown("---")
    
    if st.session_state.current_user is None:
        st.info("Please login or register using the sidebar to start chatting")
    else:
        # Display medical report mode status
        if st.session_state.medical_report_mode:
            st.warning("📝 Medical Report Mode Active - All messages are being saved to your medical record")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.chat_message("user", avatar="👤").write(message['content'])
                else:
                    st.chat_message("assistant", avatar="🏥").write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Type your health question or concern...")
        
        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            
            # Handle medical report mode
            if st.session_state.medical_report_mode:
                st.session_state.chatbot.update_medical_info(st.session_state.current_user, user_input)
                response = "✓ Added to your medical record. Continue providing information or click 'Stop Medical Report' to exit."
            else:
                # Handle commands
                if user_input.lower() == '/medicalreport':
                    st.session_state.medical_report_mode = True
                    response = "Medical report mode started. All your messages will be saved to your medical file."
                else:
                    # Generate AI response using Gemini API
                    user_data = st.session_state.chatbot.users[st.session_state.current_user]
                    intent = st.session_state.chatbot.detect_intent(user_input)
                    response = st.session_state.chatbot.generate_response(user_input, user_data, intent)
            
            # Add assistant response to chat
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
            # Rerun to update display
            st.rerun()

if __name__ == "__main__":
    main()
