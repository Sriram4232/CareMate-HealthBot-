# CareMate-HealthBot-
An Interactive Chatbot which give users nutrients plan, diet suggestions, and a symptom check.

login page:

⦁	Name
⦁	age
⦁	gender
⦁	country
⦁	height
⦁	weight
⦁	mobile number

1. User Input:
- User types or speaks a question or concern.---->Login

2. Intent Detection:
- Chatbot identifies if the request is about nutrition, stress, glucose monitoring, etc.---->yesterdays class as ref (focus 80% on the chosen topic and 20% on other chat)

3. Personalization:
- Chatbot tailors responses based on user data (e.g., health records, mood, device data).---->saves the user medical files into KB

4. Empathetic Interaction:
- Uses sentiment analysis to adapt tone and support level. -->use Hugging face transformers (distilbert-base-uncased-finetuned-sst-2-english) as example.

(Store in the KB in two ways 1st way is /register(creates a new file for the user) 2nd way is /login(access the user file for future references))




STRUCTURE OF THE MEDICAL FILES IN THE KB 
Name: 
DOB: 
Height: 
Weight: 
Gender: 
Country: 


Medical info:


Gemini API for nutrition, diet, symptom checking:

You don’t need to map the user KB to Gemini answers manually. Gemini (or similar LLM APIs) can read the content of a user’s KB file dynamically and generate responses based on it—if you pass the KB content as part of the prompt.

3️⃣ Call Gemini API

The API will process the prompt and generate an answer based on the KB content.

No manual mapping is required—just make sure the KB is in readable text form.

✅ Notes

If the KB is large, you might need to chunk it and feed only relevant parts to avoid hitting token limits.

You can implement semantic search / embeddings to pick the most relevant KB sections for each query before sending to Gemini.

Always include disclaimers if medical advice is being generated.

You only need to read and pass the KB programmatically, Gemini will handle mapping and answering.




HuggingFace Transformers:


It is a sentiment analysis model.

You give it a piece of text (a review, comment, tweet, message, etc.).

It returns whether the text is Positive or Negative, along with a confidence score.

Example Uses:

Customer feedback → Classify reviews as positive/negative.

Social media monitoring → Detect negative tweets about a brand.

Chatbots → Understand if a user is happy or upset.

Market research → Analyze sentiment trends in product discussions.

Content filtering → Flag strongly negative or toxic messages.

output = query({
    "inputs": "I like you. I love you",
})
