from flask import Flask, request, render_template
import re
import joblib
from googleapiclient.discovery import build

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('model_svc.pkl')
vectorizer = joblib.load('vect.pkl')

def get_youtube_comments(video_id, api_key, max_results):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    results = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText', maxResults=max_results).execute()

    for item in results['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    return comments

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_id = request.form['video_id']
        num_comments = int(request.form['num_comments'])
        api_key = 'AIzaSyDhtI4AaZXpgW2uN1INVzn30T8H_7XcR9o'  # Replace with your actual API key

        comments = get_youtube_comments(video_id, api_key, num_comments)
        cleaned_comments = [clean_text(comment) for comment in comments]
        tfidf_comments = vectorizer.transform(cleaned_comments)
        predictions = model.predict(tfidf_comments)

        # Count the number of positive and negative comments
        positive_count = sum(predictions)
        negative_count = len(predictions) - positive_count

        results = zip(comments, predictions)
        return render_template('results.html', results=results, positive_count=positive_count, negative_count=negative_count)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
