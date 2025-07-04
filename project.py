import os
import PyPDF2
import re
import shutil
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stop words
stop_words = set(stopwords.words('english'))

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Function to clean and tokenize text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)           # Remove all non-word characters
    text = text.lower()                       # Convert to lowercase
    words = text.split()                      # Simple whitespace-based tokenization
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(filtered_words)


# Read and process the job description
try:
    with open('job_description.txt', 'r', encoding='utf-8') as file:
        jd = clean_text(file.read())
except FileNotFoundError:
    print("❌ ERROR: job_description.txt not found. Please make sure it exists.")
    exit()

# Ensure folders exist
if not os.path.exists("resumes"):
    print("❌ ERROR: 'resumes' folder not found. Please create it and add PDFs.")
    exit()

if not os.path.exists("shortlisted"):
    os.makedirs("shortlisted")

vectorizer = TfidfVectorizer()

# Go through all PDF resumes
for filename in os.listdir('resumes'):
    if filename.endswith('.pdf'):
        filepath = os.path.join('resumes', filename)
        resume_text = extract_text_from_pdf(filepath)
        cleaned_resume = clean_text(resume_text)

        tfidf = vectorizer.fit_transform([jd, cleaned_resume])
        similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        print(f"{filename} ➜ Similarity Score: {similarity_score:.2f}")

        if similarity_score >= 0.3:
            shutil.copy(filepath, os.path.join('shortlisted', filename))
            print(f"✅ Shortlisted: {filename}")
        else:
            print(f"❌ Not shortlisted: {filename}")

