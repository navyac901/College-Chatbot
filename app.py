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
from twilio.twiml.messaging_response import MessagingResponse

# Initialize Flask app
app = Flask(__name__)

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
translator = Translator()

# OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Preprocessing
def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Prediction
def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    if not ints:
        return None
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

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

def analyze_sentiment(message):
    blob = TextBlob(message)
    return blob.sentiment.polarity

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    try:
        data = request.get_json()
        message = data.get("message")
        language = data.get("language", "en")
        original_message = message

        if not message:
            return jsonify({"reply": "No message received.", "sentiment": 0})

        if language != "en":
            try:
                message = translator.translate(message, src=language, dest='en').text
            except Exception as e:
                print(f"Translation Error: {e}")

        intents_pred = predict_class(message, model)
        response = get_response(intents_pred, intents)
        if response is None:
            response = gpt_fallback(message)

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

@app.route("/fee", methods=["POST"])
def fee_assistant():
    data = request.get_json()
    return jsonify({"usn": data.get("usn"), "amount_due": "₹10,000", "due_date": "2025-05-15"})

@app.route("/certificate", methods=["POST"])
def certificate_verification():
    data = request.get_json()
    return jsonify({
        "roll_number": data.get("roll_number"),
        "year": data.get("year"),
        "status": "Verified"
    })

@app.route("/events", methods=["GET"])
def event_notifications():
    return jsonify([
        {"title": "Tech Talk", "date": "2025-05-20"},
        {"title": "Workshop on AI", "date": "2025-05-25"}
    ])

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    return jsonify({"message": "Thank you for your feedback!"})

@app.route("/dashboard", methods=["POST"])
def student_dashboard():
    return jsonify({
        "attendance": "95%",
        "grades": {"Math": "A", "Science": "B+"},
        "schedule": {"Monday": "Math", "Tuesday": "Science"}
    })

@app.route("/whatsapp", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.form.get('Body')
    sender = request.form.get('From')
    print(f"Message from {sender}: {incoming_msg}")

    language = "en"
    response_text = "Sorry, I couldn't process that."

    if incoming_msg:
        try:
            translated_msg = translator.translate(incoming_msg, src=language, dest='en').text
            intents_pred = predict_class(translated_msg, model)
            response_text = get_response(intents_pred, intents)
            if not response_text:
                response_text = gpt_fallback(translated_msg)
            response_text = translator.translate(response_text, src='en', dest=language).text
        except Exception as e:
            print(f"WhatsApp error: {e}")

    resp = MessagingResponse()
    resp.message(response_text)
    return str(resp)

# Final entry point (Render friendly)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
