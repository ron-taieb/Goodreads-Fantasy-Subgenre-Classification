import pandas as pd
import re
from langdetect import detect, LangDetectException
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download stopwords if not present
nltk.download('stopwords')

# Load translation model
model_name = 'Helsinki-NLP/opus-mt-mul-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)  # remove short words
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def translate_text(text, model, tokenizer):
    translated = model.generate(**tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def safe_detect(text):
    if not text or not any(c.isalpha() for c in text):
        return 'en'
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def truncate_text(text, max_length=512):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    truncated_text = tokenizer.decode(tokens['input_ids'][0][:max_length], skip_special_tokens=True)
    return truncated_text

def process_reviews_subset(start_index, end_index, input_feather, output_feather):
    # Load the subset of data
    df = pd.read_feather(input_feather)
    subset_df = df.iloc[start_index:end_index].copy()

    # Process and translate the text
    subset_df['cleaned_text'] = subset_df['review_text'].apply(lambda x: translate_text(truncate_text(x), model, tokenizer) if safe_detect(x) != 'en' else truncate_text(x))
    subset_df['cleaned_text'] = subset_df['cleaned_text'].apply(preprocess)

    # Remove rows with empty cleaned text
    subset_df = subset_df[subset_df['cleaned_text'] != '']

    # Save the processed subset
    subset_df.to_feather(output_feather)

if __name__ == "__main__":
    import sys
    start_index = int(sys.argv[1])
    end_index = int(sys.argv[2])
    input_feather = sys.argv[3]
    output_feather = sys.argv[4]

    process_reviews_subset(start_index, end_index, input_feather, output_feather)
