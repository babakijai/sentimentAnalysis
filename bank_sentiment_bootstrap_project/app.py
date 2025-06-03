from flask import Flask, render_template, request
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from collections import defaultdict
from datetime import datetime
from flask import request, render_template
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import spacy
import os
import nltk
import re

app = Flask(__name__)

# Download lexicon if not already available
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

feedback_store = []
sentiment_counts = {}  # {"#hashtag": {"positive": 0, "neutral": 0, "negative": 0}}

def extract_hashtags(text):
    hashtags = re.findall(r"#\w+", text)
    normalized = [tag.lower() for tag in hashtags]  # Convert to lowercase
    return normalized

def get_sentiment(feedback):
    scores = vader.polarity_scores(feedback)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def generate_trend_data(feedback_store):
    # Prepare dictionary to count sentiments by date
    daily_counts = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})

    for fb in feedback_store:
        date = fb.get('timestamp')
        sentiment = fb.get('sentiment', '').lower()
        if sentiment in ['positive', 'neutral', 'negative']:
            daily_counts[date][sentiment] += 1

    # Sort by date
    sorted_dates = sorted(daily_counts.keys())
    trend_data = {
        'dates': sorted_dates,
        'positive': [daily_counts[date]['positive'] for date in sorted_dates],
        'neutral': [daily_counts[date]['neutral'] for date in sorted_dates],
        'negative': [daily_counts[date]['negative'] for date in sorted_dates]
    }

    return trend_data

def detect_category(text):
    text = text.lower()
    if any(word in text for word in ['bad', 'issue', 'problem', 'error', 'complaint']):
        return "Complaint"
    elif any(word in text for word in ['thanks', 'love', 'great', 'awesome', 'good']):
        return "Praise"
    elif any(word in text for word in ['should', 'could', 'suggest', 'recommend']):
        return "Suggestion"
    elif any(word in text for word in ['how', 'when', 'where', 'why']):
        return "Inquiry"
    return "General"


def generate_wordcloud(feedback_text, output_path="static/wordcloud.png"):
    wc = WordCloud(width=800, height=400, background_color='white').generate(feedback_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    wc.to_file(output_path)


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(feedback_text):
    # 1. Preprocess: combine hashtags as one word
    cleaned_text = re.sub(r'#(\w+)', r'hashtag_\1', feedback_text)

    # 2. Run NER
    doc = nlp(cleaned_text)
    
    # 3. Filter and clean entities
    entities = []
    useful_labels = {"ORG", "GPE", "PERSON"}

    for ent in doc.ents:
        ent_text = ent.text.strip()

        # Skip noisy entries
        if ent_text == "#" or (ent.label_ == "CARDINAL" and not ent_text.isdigit()):
            continue

        if ent.label_ in useful_labels:
            # Restore original hashtag format
            ent_text = ent_text.replace("hashtag_", "#")
            entities.append((ent_text, ent.label_))
    
    return entities
'''
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_entities_bert(feedback_text):
    results = ner_pipeline(feedback_text)
    entities = [(r["word"], r["entity_group"]) for r in results]
    return entities
'''

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", last_feedback=None, last_sentiment=None, stats={}, mood_score=0, mood_label="Neutral", chart_data={})

@app.route("/filter", methods=["POST"])
def filter_feedback():
    start_date_str = request.form["start_date"]
    end_date_str = request.form["end_date"]

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Filter feedback_store by date range
    filtered = [
        fb for fb in feedback_store
        if "timestamp" in fb and start_date <= datetime.strptime(fb["timestamp"], "%Y-%m-%d").date() <= end_date
    ]

    return render_template(
        "index.html",
        filtered=filtered
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    name = request.form.get("name")
    feedback = request.form.get("feedback")
    last_submited_on=datetime.now().strftime("%Y-%m-%d")

    hashtags = extract_hashtags(feedback)
    text_lower = feedback.lower()

    # Skip if 'wellsfargo' not mentioned in plain text or hashtag
    if "wellsfargo" not in text_lower and all("wellsfargo" not in tag for tag in hashtags):
        return render_template("index.html",
                               last_feedback=None,
                               last_sentiment=None,
                               stats={"total": len(feedback_store),
                                      "positive": len([f for f in feedback_store if f['sentiment'] == 'positive']),
                                      "neutral": len([f for f in feedback_store if f['sentiment'] == 'neutral']),
                                      "negative": len([f for f in feedback_store if f['sentiment'] == 'negative'])},
                               mood_score=0,
                               mood_label="Filtered (not WellsFargo)",
                               chart_data={})

    # Analyze sentiment
    sentiment = get_sentiment(feedback)
    category = detect_category(feedback)
    feedback_store.append({
        "name": name,
        "feedback": feedback,
        "sentiment": sentiment,
        "hashtags": hashtags,
        "category":category,
        "timestamp": datetime.now().strftime("%Y-%m-%d")
    })

    for tag in hashtags:
        sentiment_counts.setdefault(tag, {"positive": 0, "neutral": 0, "negative": 0})
        sentiment_counts[tag][sentiment] += 1

    # Stats summary
    total = len(feedback_store)
    pos = len([f for f in feedback_store if f["sentiment"] == "positive"])
    neu = len([f for f in feedback_store if f["sentiment"] == "neutral"])
    neg = len([f for f in feedback_store if f["sentiment"] == "negative"])

    mood_score = pos - neg
    mood_label = "Positive" if mood_score > 0 else ("Negative" if mood_score < 0 else "Neutral")

    chart_data = {
        "labels": list(sentiment_counts.keys()),
        "positive": [sentiment_counts[tag]["positive"] for tag in sentiment_counts],
        "neutral": [sentiment_counts[tag]["neutral"] for tag in sentiment_counts],
        "negative": [sentiment_counts[tag]["negative"] for tag in sentiment_counts],
    }

    # Entity Recognition
    entities = extract_entities(feedback)
    # Generate word cloud image
    all_feedback_text = " ".join(fb["feedback"] for fb in feedback_store)
    generate_wordcloud(all_feedback_text)
    trend_data = generate_trend_data(feedback_store) if feedback_store else {}
    
    return render_template("index.html",
                           last_feedback={"name": name, "feedback": feedback},
                           last_sentiment=sentiment,
                           last_submited_on=last_submited_on,
                           stats={"total": total, "positive": pos, "neutral": neu, "negative": neg},
                           mood_score=mood_score,
                           mood_label=mood_label,
                           chart_data=chart_data,
                           entities=entities,
                           trend_data=trend_data,
                           wordcloud_url='static/wordcloud.png')

if __name__ == "__main__":
    app.run(debug=True)
