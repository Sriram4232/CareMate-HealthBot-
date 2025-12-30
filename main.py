import streamlit as st
import json
import os
import requests
from datetime import datetime
import re
from transformers import pipeline
import torch
import google.generativeai as genai

# API Keys - Using only Gemini
GEMINI_API_KEY = ""

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class MedicalChatbot:
    def __init__(self):
        self.sentiment_analyzer = self.load_sentiment_analyzer()
        self.users_file = "kb/users.json"
        self.diet_file = "diet.json"
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
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
        
        self.users[phone] = {
            'Name': user_data['name'],
            'DOB': user_data['age'],
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
    
    def get_gemini_nutrition_advice(self, user_data, diet_context, user_query):
        """Get personalized nutrition advice using Gemini API"""
        try:
            # Prepare user context
            user_context = f"""
            User Profile:
            - Name: {user_data['Name']}
            - Age: {user_data['DOB']}
            - Height: {user_data['Height']} cm
            - Weight: {user_data['Weight']} kg
            - Gender: {user_data['Gender']}
            - Country: {user_data['Country']}
            
            Current Diet: {diet_context}
            User Query: {user_query}
            
            Please provide specific, practical nutrition advice. Focus on:
            1. Addressing the current diet issues
            2. Providing healthy alternatives
            3. Creating a balanced meal plan
            4. Considering the user's profile
            """
            
            response = self.gemini_model.generate_content(user_context)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble accessing nutrition advice right now. Error: {str(e)}"
    
    def get_gemini_symptom_analysis(self, symptoms_text, user_data=None):
        """Analyze symptoms using Gemini API"""
        try:
            # Prepare context for symptom analysis
            user_context = ""
            if user_data:
                user_context = f"""
                User Profile:
                - Age: {user_data['DOB']}
                - Gender: {user_data['Gender']}
                - Height: {user_data['Height']} cm
                - Weight: {user_data['Weight']} kg
                """
            
            symptom_prompt = f"""
            {user_context}
            User is describing these symptoms: {symptoms_text}
            
            Please provide:
            1. Possible common causes (but emphasize this is not a diagnosis)
            2. General self-care recommendations
            3. When to seek medical attention
            4. Important precautions
            
            IMPORTANT: Always include a disclaimer that this is not medical advice and users should consult healthcare professionals.
            Be empathetic and practical in your response.
            """
            
            response = self.gemini_model.generate_content(symptom_prompt)
            return response.text
        except Exception as e:
            return self.get_basic_symptom_advice(symptoms_text)
    
    def get_basic_symptom_advice(self, symptom_text):
        """Fallback symptom advice"""
        symptom_lower = symptom_text.lower()
        
        advice_map = {
            'fever': "Rest, stay hydrated, and monitor temperature. Consult a doctor if fever persists above 38°C (100.4°F) for more than 3 days.",
            'headache': "Rest in a quiet room, stay hydrated, avoid bright screens. Seek medical advice if severe or accompanied by other symptoms.",
            'cough': "Stay hydrated, use honey in warm water, avoid irritants. See a doctor if accompanied by fever or breathing difficulties.",
            'stomach': "Eat bland foods, avoid spicy/greasy foods, stay hydrated. Seek medical attention if severe pain or vomiting persists.",
            'pain': "Rest the affected area, apply cold/warm compress as appropriate. Consult a doctor if pain is severe or worsening.",
            'nausea': "Sip clear fluids, eat small bland meals, avoid strong odors. Seek help if unable to keep fluids down.",
            'dizziness': "Sit or lie down immediately, rise slowly from sitting position. Consult doctor if frequent or severe."
        }
        
        for symptom, advice in advice_map.items():
            if symptom in symptom_lower:
                return f"I understand you're experiencing {symptom}. {advice} Please consult a healthcare professional for proper diagnosis and treatment."
        
        return "I understand you're not feeling well. It's important to monitor your symptoms and consult a healthcare professional if they persist or worsen. Rest and stay hydrated in the meantime."

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
        
        # Intent-based responses using Gemini API
        if intent == 'nutrition' and user_data:
            # Update diet history
            self.update_diet_history(st.session_state.current_user, user_input)
            
            # Get Gemini advice
            gemini_response = self.get_gemini_nutrition_advice(user_data, user_input, user_input)
            
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
            gemini_response = self.get_gemini_symptom_analysis(user_input, user_data)
            return empathetic_opening + gemini_response
        
        elif intent == 'mental_health':
            # Use Gemini for mental health advice
            try:
                mental_health_prompt = f"""
                User is seeking mental health advice. They mentioned: {user_input}
                Provide supportive, practical advice for stress/anxiety management.
                Focus on: breathing exercises, routine, sleep, and when to seek professional help.
                Be empathetic and non-judgmental.
                """
                response = self.gemini_model.generate_content(mental_health_prompt)
                return empathetic_opening + response.text
            except:
                basic_advice = "Consider practicing relaxation techniques, maintaining a regular routine, and ensuring adequate sleep. If feelings persist, speaking with a mental health professional can be very helpful."
                return empathetic_opening + basic_advice
        
        elif intent == 'fitness' and user_data:
            # Use Gemini for fitness advice
            try:
                fitness_prompt = f"""
                User profile: 
                - Age: {user_data.get('DOB', 'Not specified')}
                - Gender: {user_data.get('Gender', 'Not specified')}
                - Height: {user_data.get('Height', 'Not specified')} cm
                - Weight: {user_data.get('Weight', 'Not specified')} kg
                
                Query: {user_input}
                
                Provide personalized fitness advice considering age, weight, and goals.
                Include practical exercise recommendations and safety precautions.
                """
                response = self.gemini_model.generate_content(fitness_prompt)
                return empathetic_opening + response.text
            except:
                try:
                    height_m = float(user_data['Height']) / 100
                    weight_kg = float(user_data['Weight'])
                    bmi = weight_kg / (height_m ** 2)
                    bmi_advice = f"Your BMI is {bmi:.1f}. "
                    if bmi < 18.5:
                        bmi_advice += "Focus on strength training and balanced nutrition."
                    elif bmi <= 25:
                        bmi_advice += "Maintain with regular cardio and strength exercises."
                    else:
                        bmi_advice += "Combine cardio with strength training and balanced diet."
                except:
                    bmi_advice = "Regular exercise and balanced nutrition are key."
                
                return empathetic_opening + bmi_advice
        
        else:
            # General health advice using Gemini
            try:
                general_prompt = f"User asked: {user_input}. Provide helpful, practical health advice. If it's a greeting, respond appropriately."
                response = self.gemini_model.generate_content(general_prompt)
                return empathetic_opening + response.text
            except:
                return empathetic_opening + "How can I assist you with your health concerns today? You can ask me about nutrition, symptoms, fitness, or general health advice."

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
        
        # User info display
        st.sidebar.subheader("User Information")
        st.sidebar.write(f"*Name:* {user_data['Name']}")
        st.sidebar.write(f"*Age:* {user_data['DOB']}")
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

