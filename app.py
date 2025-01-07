import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # If needed

from flask import Flask, render_template, request
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Load summarization model (T5 model from Hugging Face)
summarizer = pipeline('summarization', model='google-t5/t5-small')

# Hybrid Summarization Function
def hybrid_text_summarizer(text):
    sentences = sent_tokenize(text)
    chunk_size = 5  # Number of sentences per chunk for summarization
    summarized_chunks = []

    # Split text into chunks of sentences
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size])
        summarized_chunk = summarizer(chunk)
        summarized_chunks.append(summarized_chunk[0]['summary_text'])

    # Combine all summarized chunks into a final summary
    final_summary = " ".join(summarized_chunks)
    return final_summary

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        summary = hybrid_text_summarizer(text)
        return render_template("index.html", summary=summary)
    return render_template("index.html", summary="")

if __name__ == "__main__":
    app.run(debug=True)
