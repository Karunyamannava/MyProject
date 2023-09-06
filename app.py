from flask import Flask, render_template, request, redirect, url_for
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import io
import base64
import nltk
import random

nltk.download('vader_lexicon')

app = Flask(__name__)

app.jinja_env.globals.update(enumerate=enumerate, zip=zip)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        return redirect(url_for('result', url=url))
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        url = request.form['url']

        # get YouTube URL from form input
        video_id = url.split('=')[1]

        # create YouTube API client
        youtube = build('youtube', 'v3', developerKey='AIzaSyCqozWcXN7tQdlUYjDjDjngijvLKks-BE4')

        # get comments from YouTube video
        comments = []
        nextPageToken = None
        while True:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=nextPageToken,
                textFormat='plainText'
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in response:
                nextPageToken = response['nextPageToken']
            else:
                break

        # perform sentiment analysis on comments
        sia = SentimentIntensityAnalyzer()
        svm = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
        X = comments
        y = [1 if sia.polarity_scores(comment)['compound'] > 0 else 0 for comment in comments]
        split_idx = int(0.8 * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        svm.fit(X_train, y_train)

        # get polarity scores for all comments
        polarity_scores = [sia.polarity_scores(comment)['compound'] for comment in comments]

        # set threshold dynamically as the median of positive polarity scores
        positive_scores = [score for score in polarity_scores if score > 0]
        threshold = sorted(positive_scores)[len(positive_scores)//2] if positive_scores else 0

        # classify comments based on the dynamic threshold
        y_pred = [1 if score > threshold else 0 for score in polarity_scores]

        # post-processing: consider specific phrases or words indicating positive sentiment
        for i, comment in enumerate(comments):
            if any(word in comment for word in ['cool', 'awesome', 'great', 'good']):
                y_pred[i] = 1

        # calculate accuracy and F1 scores
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

         # get top 5 positive comments based on SVM predictions and their polarity scores
        positive_comments = [(comment, '+', sia.polarity_scores(comment)['compound']) for score, comment in
                            sorted(zip(svm.predict_proba(X)[:, 1], comments), reverse=True) if svm.predict([comment])[0] == 1 and sia.polarity_scores(comment)['compound'] > threshold][:5]

        # get top 5 negative comments based on SVM predictions and their polarity scores
        negative_comments = [(comment, '-', sia.polarity_scores(comment)['compound']) for score, comment in
                            sorted(zip(svm.predict_proba(X)[:, 1], comments)) if svm.predict([comment])[0] == 0 and sia.polarity_scores(comment)['compound'] < -threshold][:5]

        # create pie chart
        positive = len([comment for comment in comments if svm.predict_proba([comment])[0][1] > 0.7])
        negative = len([comment for comment in comments if svm.predict_proba([comment])[0][1] < 0.3])
        neutral = len(comments) - positive - negative
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [positive, negative, neutral]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        explode = (0.1, 0.1, 0.1)  # explode positive, negative, and neutral slices
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Sentiment Analysis of Comments')
        # convert plot to base64 image for embedding in HTML
        img = io.BytesIO()
        plt.savefig('static/sentiment.png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # calculate F1 scores
        total = positive + negative + neutral
        f1_positive = 2 * positive / (2 * positive + negative + neutral)
        f1_negative = 2 * negative / (2 * negative + positive + neutral)
        f1_neutral = 2 * neutral / (2 * neutral + positive + negative)

        # render template with results
        return render_template('result.html', url=url, positive_comments=positive_comments, negative_comments=negative_comments, 
                               plot_url=plot_url, total=total, negative=negative, neutral=neutral, f1_positive=f1_positive, 
                               f1_negative=f1_negative, f1_neutral=f1_neutral, accuracy=accuracy, f1=f1, positive=positive)

    # Redirect to index if accessed directly
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
