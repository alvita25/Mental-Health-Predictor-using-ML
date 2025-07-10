



import os
import re
import pickle
import numpy as np
import nltk
import requests
import cohere
import google.generativeai as genai

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from dotenv import load_dotenv

from models import db, bcrypt, User, Prediction

load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
bcrypt.init_app(app)

# AI service keys
co = cohere.Client(os.getenv("COHERE_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load ML models
with open("xgboost_model.pkl", "rb") as f: xgb_model = pickle.load(f)
with open("lgbm_model.pkl", "rb") as f: lgbm_model = pickle.load(f)
with open("best_logistic_regression_model.pkl", "rb") as f: logistic_model = pickle.load(f)
with open("best_bernoulli_nb_model.pkl", "rb") as f: nb_model = pickle.load(f)
with open("random_forest_model.pkl", "rb") as f: random_forest_model = pickle.load(f)
# Load vectorizers  
with open("tfidf_vectorizer.pkl", "rb") as f: vector_default = pickle.load(f)
with open("xgb_tfidf_vectorizer.pkl", "rb") as f: vector_xgb = pickle.load(f)
with open("tfidf_vectorizer_logistic.pkl", "rb") as f: vector_logistic = pickle.load(f)
with open("tfidf_vectorizer_nb.pkl", "rb") as f: vector_nb = pickle.load(f)
with open("random_forest_tfidf_vectorizer.pkl", "rb") as f: vector_random_forest = pickle.load(f)

#encoders
with open("label_encoder.pkl", "rb") as f: label_enc = pickle.load(f)
with open("xgb_label_encoder.pkl", "rb") as f: label_enc_xgb = pickle.load(f)
with open("label_encoder_nb.pkl", "rb") as f: label_enc_nb = pickle.load(f)
with open("random_forest_label_encoder.pkl", "rb") as f: label_enc_random_forest = pickle.load(f)
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))

def preprocess_text(text, model_type="default"):
    original = text
    t = text.lower()
    t = re.sub(r'http[s]?://\S+|\[.*?\]\(.*?\)|@\w+|[^\w\s]', "", t).strip()
    tokens = word_tokenize(t)
    stemmed = " ".join(lemmatizer.lemmatize(w, pos="v") for w in tokens)
    clean = " ".join(w for w in stemmed.split() if w not in stops)

    num_chars = len(original)
    num_sentences = len(nltk.sent_tokenize(original))

    if model_type == "xgb":
        vect = vector_xgb
    elif model_type == "logistic":
        vect = vector_logistic
    elif model_type == "nb":
        vect = vector_nb
    elif model_type == "random_forest":
        vect = vector_random_forest
    else:
        vect = vector_default

    tfidf = vect.transform([clean])
    return hstack([tfidf, np.array([[num_chars, num_sentences]])])



def ai_response(native_text, lang_code):
    prompt = f"User speaks in {lang_code}. Reply empathetically in that language:\nUser: {native_text}\nResponse:"
    try:
        resp = gemini_model.generate_content(prompt)
        print("🤖 Gemini full response:", resp)
        return resp.text.strip() if hasattr(resp, 'text') else str(resp)
    except Exception as e:
        print("🔥 Gemini Error:", e)
        return "Sorry, I’m having trouble responding right now."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name, email, pw = request.form['name'], request.form['email'], request.form['password']
        if User.query.filter_by(email=email).first():
            flash("Email already exists", "error")
            return redirect(url_for('login'))
        hashed = bcrypt.generate_password_hash(pw).decode('utf-8')
        db.session.add(User(name=name, email=email, password_hash=hashed))
        db.session.commit()
        flash("Registered! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email, pw = request.form['email'], request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user or not bcrypt.check_password_hash(user.password_hash, pw):
            flash("Invalid credentials", "error")
            return redirect(url_for('login'))
        session['user_id'] = user.id
        flash("Logged in!", "success")
        return redirect(url_for('profile'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for('home'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash("Please log in", "warning")
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        native = data.get('native', "")
        lang = data.get('language', "en-US")

        print("✅ Received data:", data)
        print("🌐 Language:", lang)
        print("🧠 Native input:", native)

        # TEMP: Replace with dummy reply for now
        # reply = ai_response(native, lang)
        reply = ai_response(native, lang)


        return jsonify({
            "native": native,
            "language": lang,
            "response_native": reply,
            "translated": data.get("translated", ""),
            "response": None
        })

    except Exception as e:
        print("🔥 ERROR in /chat route:", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['chat_history']

    # Preprocess and predict for each model
    feats_xgb = preprocess_text(text, "xgb")
    feats_log = preprocess_text(text, "logistic")
    feats_nb = preprocess_text(text, "nb")
    feats_rf = preprocess_text(text, "random_forest")

    # XGBoost
    pred_xgb = xgb_model.predict_proba(feats_xgb)[0]
    xgb_index = np.argmax(pred_xgb)
    rx = label_enc_xgb.inverse_transform([xgb_index])[0]
    xgb_certainty = round(pred_xgb[xgb_index] * 100, 2)

    # LightGBM
    pred_lgbm = lgbm_model.predict_proba(feats_xgb)[0]
    lgbm_index = np.argmax(pred_lgbm)
    rl = label_enc_xgb.inverse_transform([lgbm_index])[0]
    lgbm_certainty = round(pred_lgbm[lgbm_index] * 100, 2)

    # Logistic Regression
    pred_log = logistic_model.predict_proba(feats_log)[0]
    log_index = np.argmax(pred_log)
    rlog = label_enc.inverse_transform([log_index])[0]
    log_certainty = round(pred_log[log_index] * 100, 2)

    # Naive Bayes
    pred_nb = nb_model.predict_proba(feats_nb)[0]
    nb_index = np.argmax(pred_nb)
    rnb = label_enc_nb.inverse_transform([nb_index])[0]
    nb_certainty = round(pred_nb[nb_index] * 100, 2)

    # Random Forest
    pred_rf = random_forest_model.predict_proba(feats_rf)[0]
    rf_index = np.argmax(pred_rf)
    rrf = label_enc_random_forest.inverse_transform([rf_index])[0]
    rf_certainty = round(pred_rf[rf_index] * 100, 2)

    # Save prediction to database if logged in
    if 'user_id' in session:
      user = User.query.get(session['user_id'])
      db.session.add(Prediction(
        result_xgb=rx,
        certainty_xgb=xgb_certainty,
        result_lgbm=rl,
        certainty_lgbm=lgbm_certainty,
        result_logistic=rlog,
        certainty_logistic=log_certainty,
        result_nb=rnb,
        certainty_nb=nb_certainty,
        result_rf=rrf,
        certainty_rf=rf_certainty,
        user_id=user.id
       ))
      db.session.commit()

    user = User.query.get(session['user_id']) if 'user_id' in session else None

    return render_template('profile.html',
        xgb_result=rx, xgb_certainty=xgb_certainty,
        lgbm_result=rl, lgbm_certainty=lgbm_certainty,
        logistic_result=rlog, log_certainty=log_certainty,
        nb_result=rnb, nb_certainty=nb_certainty,
        rf_result=rrf, rf_certainty=rf_certainty,
        chat_history_display=text,
        user=user)


@app.route('/history')
def history():
    if 'user_id' not in session:
        flash("Login required", "warning")
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('history.html', predictions=user.predictions, user=user)

if __name__ == '__main__':
    app.run(debug=True)
