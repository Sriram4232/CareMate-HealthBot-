🏥 CareMate – HealthBot

CareMate is an interactive **medical health chatbot** built using **Streamlit**, **Google Gemini API**, and **Hugging Face Transformers**.
It helps users with **nutrition guidance**, **diet suggestions**, **symptom checking**, **fitness advice**, and **basic mental health support**.

⚠️ *This chatbot provides general health information only and is not a substitute for professional medical advice.*

---

✨ Features

 🔐 User Login & Registration

Users can register and log in by providing:

* Name
* Age
* Gender
* Country
* Height
* Weight
* Mobile Number

Each user has a personalized medical profile.

---

 🧠 Intent Detection

The chatbot identifies the user’s intent and classifies queries into:

* Nutrition / Diet
* Symptoms
* Mental Health
* Fitness
* General Health

Responses focus mainly on the detected intent.

---

 ❤️ Sentiment Analysis

Uses **Hugging Face Transformers**
Model: `distilbert-base-uncased-finetuned-sst-2-english`

* Detects positive or negative sentiment
* Adjusts the response tone to be empathetic and supportive

---

 📂 Medical Knowledge Base (KB)

User information and medical notes are stored locally.

**Medical record includes:**

* Name
* Age
* Height
* Weight
* Gender
* Country
* Medical Notes
* Diet History

Users can enable **Medical Report Mode** to save messages directly into their medical record.

---

 🥗 Nutrition & Diet Guidance

* Analyzes diet-related input
* Identifies unhealthy food patterns
* Suggests healthier alternatives
* Provides personalized advice based on user details
* Displays BMI for informational purposes only

---

 🤒 Symptom Checker

* Provides possible common causes (non-diagnostic)
* Suggests general self-care practices
* Advises when to consult a healthcare professional
* Does not provide diagnosis or prescriptions

---

 🧠 Mental Health Support

* Handles stress and mood-related queries
* Offers general coping strategies
* Encourages professional help when needed
* Uses supportive and non-judgmental language

---

 🤖 Gemini API Integration

CareMate uses **Google Gemini** to generate responses for:

* Nutrition advice
* Symptom explanations
* Fitness guidance
* Mental health support

User medical data is passed as readable context to the model to generate personalized responses.

---

🛠️ Technologies Used

* Python
* Streamlit
* Google Gemini API
* Hugging Face Transformers
* JSON-based Knowledge Base

---

⚠️ Disclaimer

CareMate provides **general health information only**.
It does **not diagnose diseases**, **does not prescribe medication**, and **does not replace professional medical consultation**.

---

🎓 Purpose

This project is developed for **educational and academic purposes.
