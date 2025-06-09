from flask import Flask, render_template, request, render_template,flash
from nltk.sentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename
from wordcloud import WordCloud
from collections import defaultdict
from datetime import datetime
from transformers import pipeline
import matplotlib.pyplot as plt
import spacy
import pandas as pd
import os
import nltk
import re

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Download lexicon if not already available
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

feedback_store = []
sentiment_counts = {}  # {"#hashtag": {"positive": 0, "neutral": 0, "negative": 0}}

# Load only once to avoid reloading on every call
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Predefined categories
CATEGORIES = ["Complaint", "Praise", "Suggestion", "Inquiry", "General"]

def detect_category(text, threshold=0.5):
    """
    Categorizes text using zero-shot classification.

    Args:
        text (str): The input user message.
        threshold (float): Confidence threshold to trust prediction.

    Returns:
        str: One of "Complaint", "Praise", "Suggestion", "Inquiry", "General"
    """
    if not text.strip():
        return "General"

    result = classifier(text, CATEGORIES)
    top_label = result['labels'][0]
    top_score = result['scores'][0]

    if top_score < threshold:
        return "General"
    return top_label


def extract_hashtags(text):
    hashtags = re.findall(r"#\w+(?:_\w+)*", text)
    normalized = [tag.lower() for tag in hashtags]
    
    # Check if any of the hashtags is wells_fargo or wellsfargo (case-insensitive)
    if any(tag in ['#wellsfargo', '#wells_fargo'] for tag in normalized):
        return normalized
    else:
        return []

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
    complaint_keywords = [
    'bad', 'issue', 'problem', 'error', 'complaint', 'worst', 'not working',
    'declined', 'failed', 'unable', 'delay', 'disappointed', 'frustrated',
    'charged wrongly', 'scam', 'unauthorized', 'lost money', 'refund not received',
    'rude', 'blocked', 'poor service', 'unhappy', 'cheated', 'angry', 'irresponsible'
]
    praise_keywords = [
    'thank you', 'thanks', 'love', 'great', 'awesome', 'excellent',
    'amazing', 'satisfied', 'happy', 'appreciate', 'well done', 'good service',
    'fast response', 'supportive', 'quick resolution', 'smooth experience'
]

    suggestion_keywords = [
    'should', 'could', 'would be better', 'suggest', 'recommend',
    'wish', 'hope you can', 'improve', 'consider adding', 'needs improvement',
    'better if', 'please add', 'feature request'
]
    inquiry_keywords = [
    'how to', 'how do I', 'when will', 'where is', 'why', 'what if',
    'can you', 'need help', 'unable to understand', 'question', 'clarify',
    'explain', 'who do I contact', 'want to know'
]

    text = text.lower()
    if match_keywords(text, complaint_keywords):
        return "Complaint"
    elif match_keywords(text, praise_keywords):
        return "Praise"
    elif match_keywords(text, suggestion_keywords):
        return "Suggestion"
    elif match_keywords(text, inquiry_keywords):
        return "Inquiry"
    return "General"


def match_keywords(text, keywords):
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keywords)



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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_summary(stats, mood_percentage, mood_label, sentiment_counts, trend_data):
    summary = []

    # 1. Market Mood and Score
    summary.append(
        f"As of the latest analysis, the overall market sentiment is categorized as "
        f"<strong>{mood_label}</strong>, with a sentiment score of <strong>{round(mood_percentage, 2)}%</strong> based on user feedback."
    )

    # 2. Sentiment Distribution (Pie chart data)
    total = stats['total']
    pos_pct = (stats['positive'] / total) * 100 if total else 0
    neu_pct = (stats['neutral'] / total) * 100 if total else 0
    neg_pct = (stats['negative'] / total) * 100 if total else 0

    summary.append(
        f"The sentiment distribution is as follows: "
        f"<strong>{round(pos_pct, 1)}%</strong> positive, "
        f"<strong>{round(neu_pct, 1)}%</strong> neutral, and "
        f"<strong>{round(neg_pct, 1)}%</strong> negative."
    )

    # 3. Sentiment Count by Hashtag (Bar graph)
    if sentiment_counts:
        summary.append("Sentiment counts across key hashtags are summarized below:")
        tag_lines = []
        for tag, counts in sentiment_counts.items():
            tag_lines.append(
                f"<strong>{tag}</strong>: Positive: {counts['positive']}, Neutral: {counts['neutral']}, Negative: {counts['negative']}"
            )
        summary.append("<br>".join(tag_lines))

    # Feedback Trend Over Time
    trend_lines = []
    for date, pos, neu, neg in zip(
        trend_data["dates"],
        trend_data["positive"],
        trend_data["neutral"],
        trend_data["negative"]
    ):
        trend_lines.append(
            f"<li><strong>{date}</strong> — Positive: {pos}, Neutral: {neu}, Negative: {neg}</li>"
        )


    summary.append("<strong>Feedback Volume Trend (Last 3 Days):</strong><br><ul>" + "".join(trend_lines) + "</ul>")

    return "<br><br>".join(summary)


def contains_wells_fargo(text):
    text = text.lower().replace(" ", "").replace("_", "")
    return "wellsfargo" in text

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
    if not contains_wells_fargo(text) and all(not contains_wells_fargo(tag) for tag in hashtags):
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

    non_neutral_total = pos + neg
    mood_score = pos - neg
    mood_percentage = (mood_score / non_neutral_total) * 100 if non_neutral_total > 0 else 0
    #mood_score = pos - neg
    #mood_label = "Positive" if mood_score > 0 else ("Negative" if mood_score < 0 else "Neutral")
    mood_label = "Positive" if mood_percentage > 0 else ("Negative" if mood_percentage < 0 else "Neutral")

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
    stats={"total": total, "positive": pos, "neutral": neu, "negative": neg}
    summary_text = generate_summary(stats, mood_percentage, mood_label, sentiment_counts, trend_data)
    
    return render_template("index.html",
                           last_feedback={"name": name, "feedback": feedback},
                           last_sentiment=sentiment,
                           last_submited_on=last_submited_on,
                           stats=stats,
                           mood_score=round(mood_percentage, 2),
                           mood_label=mood_label,
                           chart_data=chart_data,
                           entities=entities,
                           summary_text=summary_text,
                           trend_data=trend_data,
                           wordcloud_url='static/wordcloud.png')

@app.route('/import', methods=['POST'])
def import_feedback():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.referrer)

    file = request.files['file']
    print(file.filename)
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.referrer)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        print("Received file:", filename)
        file.save(filepath)

        df = pd.read_excel(filepath) if filename.endswith('.xlsx') else pd.read_csv(filepath)
        if not {'Name', 'Feedback', 'Date'}.issubset(df.columns):
            flash("Excel must contain 'Name', 'Feedback', and 'Date' columns.", "danger")
            return redirect(request.referrer)

        sentiment_counts = {}

        for _, row in df.iterrows():
            name = row['Name']
            feedback = row['Feedback']
            date = row['Date']
            sentiment = get_sentiment(feedback)
            hashtags = extract_hashtags(feedback)
            category = detect_category(feedback)
        
            feedback_store.append({
                "name": name,
                "feedback": feedback,
                "sentiment": sentiment,
                "hashtags": hashtags,
                "category": category,
                "timestamp": pd.to_datetime(date).strftime("%Y-%m-%d") if pd.notnull(date) else datetime.now().strftime("%Y-%m-%d")
            })
        
        for tag in hashtags:
                sentiment_counts.setdefault(tag, {"positive": 0, "neutral": 0, "negative": 0})
                sentiment_counts[tag][sentiment] += 1

        # Extract all timestamps and convert to datetime objects
        dates = [datetime.strptime(f['timestamp'], "%Y-%m-%d") for f in feedback_store if 'timestamp' in f]
        # Get the latest date
        last_submitted_date = max(dates)
        last_submitted_str = last_submitted_date.strftime("%Y-%m-%d")
        last_feedbacks = [f for f in feedback_store if f['timestamp'] == last_submitted_str]
        last_submitted_str = last_submitted_date.strftime("%Y-%m-%d")
        last_feedbacks = [f for f in feedback_store if f['timestamp'] == last_submitted_str]
        last_feedback = last_feedbacks[-1] if last_feedbacks else None

        
        # Stats summary
        total = len(feedback_store)
        pos = len([f for f in feedback_store if f["sentiment"] == "positive"])
        neu = len([f for f in feedback_store if f["sentiment"] == "neutral"])
        neg = len([f for f in feedback_store if f["sentiment"] == "negative"])
        
        non_neutral_total = pos + neg
        mood_score = pos - neg
        mood_percentage = (mood_score / non_neutral_total) * 100 if non_neutral_total > 0 else 0
        #mood_score = pos - neg
        #mood_label = "Positive" if mood_score > 0 else ("Negative" if mood_score < 0 else "Neutral")
        mood_label = "Positive" if mood_percentage > 0 else ("Negative" if mood_percentage < 0 else "Neutral")
        
        chart_data = {
            "labels": list(sentiment_counts.keys()),
            "positive": [sentiment_counts[tag]["positive"] for tag in sentiment_counts],
            "neutral": [sentiment_counts[tag]["neutral"] for tag in sentiment_counts],
            "negative": [sentiment_counts[tag]["negative"] for tag in sentiment_counts],
        }
        
        entities = extract_entities(" ".join(f["feedback"] for f in feedback_store))
        generate_wordcloud(" ".join(f["feedback"] for f in feedback_store))
        trend_data = generate_trend_data(feedback_store) if feedback_store else {}
        stats={"total": total, "positive": pos, "neutral": neu, "negative": neg}
        summary_text = generate_summary(stats, mood_percentage, mood_label, sentiment_counts, trend_data)
      
        return render_template("index.html",
                               last_feedback={"name": last_feedback['name'], "feedback": last_feedback['feedback']},
                               last_sentiment=last_feedback['sentiment'],
                               last_submited_on=last_feedback['timestamp'],
                               stats=stats,
                               mood_score=round(mood_percentage, 2),
                               mood_label=mood_label,
                               chart_data=chart_data,
                               entities=entities,
                               trend_data=trend_data,
                               summary_text=summary_text,
                               wordcloud_url='static/wordcloud.png')

    # If file not allowed
    flash('Invalid file type. Please upload .xlsx or .csv files only.', 'danger')
    return redirect(request.referrer)


if __name__ == "__main__":
    app.run(debug=True)
