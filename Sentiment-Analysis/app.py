from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap 


from textblob import TextBlob, Word 
import random 
import time

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    start = time.time()

    if request.method == 'POST':
        rawtext = request.form['rawtext']

        
        blob = TextBlob(rawtext)
        received_text2 = blob
        blob_sentiment = blob.sentiment.polarity
        blob_subjectivity = blob.sentiment.subjectivity
        number_of_tokens = len(list(blob.words))

        
        if blob_sentiment > 0:
            tone = "Positive"
        elif blob_sentiment < 0:
            tone = "Negative"
        else:
            tone = "Neutral"

        
        nouns = [
            word.lemmatize()
            for word, tag in blob.tags
            if tag.startswith("NN") and word.isalpha()
        ]

        
        if len(nouns) > 0:
            rand_words = random.sample(nouns, len(nouns))
            summary = rand_words
        else:
            summary = []

        end = time.time()
        final_time = end - start

        return render_template(
            'index.html',
            received_text=received_text2,
            number_of_tokens=number_of_tokens,
            blob_sentiment=blob_sentiment,
            blob_subjectivity=blob_subjectivity,
            summary=summary,
            final_time=final_time,
            tone=tone
        )

if __name__ == '__main__':
    app.run(debug=True)
