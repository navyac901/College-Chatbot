import os
import json
import random
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
import openai
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)

# Load trained model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize lemmatizer and translator
lemmatizer = WordNetLemmatizer()
translator = Translator()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Text preprocessing
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response from intents
def get_response(ints, intents_json):
    if not ints:
        return None
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

# GPT fallback
def gpt_fallback(message):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=message,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return "I'm sorry, I couldn't understand that."

# Sentiment analysis
def analyze_sentiment(message):
    blob = TextBlob(message)
    return blob.sentiment.polarity

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chatbot response route
@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        data = request.get_json()
        message = data.get("message")
        language = data.get("language", "en")

        if not message:
            return jsonify({"reply": "No message received.", "sentiment": 0})

        original_message = message

        # Translate message to English
        if language != "en":
            try:
                message = translator.translate(message, src=language, dest='en').text
            except Exception as e:
                print(f"Translation Error: {e}")

        # Predict intent or fallback
        intents_pred = predict_class(message, model)
        response = get_response(intents_pred, intents)

        if response is None:
            response = gpt_fallback(message)

        # Translate back to original language
        if language != "en":
            try:
                response = translator.translate(response, src='en', dest=language).text
            except Exception as e:
                print(f"Translation Error: {e}")

        sentiment = analyze_sentiment(original_message)

        return jsonify({"reply": response, "sentiment": sentiment})

    except Exception as e:
        print(f"Error in /get: {e}")
        return jsonify({"reply": "Something went wrong.", "sentiment": 0})

# Fee Assistant
@app.route("/fee", methods=["POST"])
def fee_assistant():
    data = request.get_json()
    usn = data.get("usn")
    fee_details = {"usn": usn, "amount_due": "₹10,000", "due_date": "2025-05-15"}
    return jsonify(fee_details)

# Certificate Verification
@app.route("/certificate", methods=["POST"])
def certificate_verification():
    data = request.get_json()
    roll_number = data.get("roll_number")
    year = data.get("year")
    verification_status = {"roll_number": roll_number, "year": year, "status": "Verified"}
    return jsonify(verification_status)

# Event Notifications
@app.route("/events", methods=["GET"])
def event_notifications():
    events = [
        {"title": "Tech Talk", "date": "2025-05-20"},
        {"title": "Workshop on AI", "date": "2025-05-25"}
    ]
    return jsonify(events)

# Feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    name = data.get("name")
    rating = data.get("rating")
    comments = data.get("comments")
    return jsonify({"message": "Thank you for your feedback!"})

# Student Dashboard
@app.route("/dashboard", methods=["POST"])
def student_dashboard():
    data = request.get_json()
    student_id = data.get("student_id")
    dashboard_data = {
        "attendance": "95%",
        "grades": {"Math": "A", "Science": "B+"},
        "schedule": {"Monday": "Math", "Tuesday": "Science"}
    }
    return jsonify(dashboard_data)
from twilio.twiml.messaging_response import MessagingResponse

# WhatsApp webhook route
@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.form.get('Body')
    sender = request.form.get('From')
    print(f"Message from {sender}: {incoming_msg}")

    response_text = ""

    if incoming_msg:
        try:
            # Language default: English
            language = "en"
            translated_msg = translator.translate(incoming_msg, src=language, dest='en').text

            # Predict intent or fallback
            intents_pred = predict_class(translated_msg, model)
            response_text = get_response(intents_pred, intents)

            if not response_text:
                response_text = gpt_fallback(translated_msg)

            # Translate back (if you wish to detect lang automatically, integrate langdetect or similar)
            response_text = translator.translate(response_text, src='en', dest=language).text

        except Exception as e:
            print(f"WhatsApp error: {e}")
            response_text = "Sorry, I couldn't process that."

    # Build WhatsApp response
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(response_text)
    return str(resp)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
