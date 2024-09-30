# NMF Model for Topic Modeling
# Define your directory
dir = "/path/to/your/dataset"
model_dir = "/path/to/your/models"
output_dir = "/path/to/your/output"

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords and tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Load the preprocessed DataFrame
df = pd.read_feather(os.path.join(dir, "processed_reviews.feather"))

# Preprocess the text
print("Preprocessing text...")
texts = df['cleaned_text'].tolist()

# Create TF-IDF vectors
print("Creating TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=100000, max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Perform NMF
print("Performing NMF...")
nmf_model = NMF(n_components=10, random_state=1, init='nndsvd', max_iter=200)
nmf_topic_matrix = nmf_model.fit_transform(tfidf_matrix)

# Save the NMF model
nmf_model_path = os.path.join(model_dir, "nmf_model.pkl")
with open(nmf_model_path, 'wb') as model_file:
    pickle.dump(nmf_model, model_file)

# Save the TF-IDF vectorizer
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("NMF model and TF-IDF vectorizer saved.")

# Extract top 10 words for each topic
feature_names = vectorizer.get_feature_names_out()
top_10_words = {}
for topic_idx, topic in enumerate(nmf_model.components_):
    top_10_indices = topic.argsort()[-10:][::-1]
    top_10_words[f"Topic {topic_idx}"] = [feature_names[i] for i in top_10_indices]

# Assuming 'subgenre_dict' is predefined
subgenre_dict = {
    "Slice of Life": ["time", "life", "character", "people", "novel", "make", "end", "plot", "family", "story", "heart", "school", "friends", "parents"],
    "Romantic Adventure": ["stars", "romance", "adventure", "trilogy", "wonderful", "entertaining", "definitely", "ending", "love", "wait", "finish", "glad", "start", "got", "fun", "forward", "romantic", "couple", "kiss", "emotions"],
    "Literary Collections": ["reading", "collection", "novels", "style", "harry", "world", "potter", "copy", "short", "fantasy", "myth", "legend", "tale", "gods", "heroes", "quest", "epic", "mythology", "saga", "adventure", "written", "author"],
    "Suspense and Mystery": ["wait", "finish", "slow", "glad", "end", "start", "got", "review", "mystery", "plot", "character", "bad", "lot", "thought", "people", "make", "times", "pages", "thriller", "detective", "crime", "suspense"],
    "Paranormal Fantasy": ["magic", "vampires", "fairy", "novel", "family", "girl", "want", "character", "time", "hot", "vampire", "romance", "paranormal", "supernatural", "blood", "zombie", "werewolf", "ghost", "hunter", "wizard", "witch", "curse"],
    "Quick Reads": ["pages", "enjoyed", "slow", "forward", "short", "end", "chapters", "plot", "started", "boring", "quick", "light", "fun", "stories", "cute", "entertaining", "easy", "fast", "interesting", "read", "bite-sized"],
    "Series and Sequels": ["series", "trilogy", "sequel", "installment", "prequel", "volume", "chapter", "continuation", "cliffhanger", "finale", "saga", "continuation", "part", "ongoing", "installments", "follow-up", "episode", "next", "series", "book"],
    "Epic Fantasy": ["king", "queen", "prince", "dragon", "warrior", "throne", "kingdom", "battle", "sword", "heroic", "quest", "epic", "myth", "legend", "adventure", "gods", "heroes", "kingdom", "magic", "power"],
    "Angels, Demons, and Sci-Fi": ["angel", "relationship", "demon", "mortal", "heaven", "hell", "fallen", "soul", "curse", "dark", "world", "universe", "dimension", "realm", "alternate", "space", "galaxy", "future", "alien", "planet", "technology", "spacecraft", "extraterrestrial", "cyber", "dystopia", "futuristic", "time", "travel", "robots", "parallel"],
    "Opinion about review": ["amazing", "ending", "absolutely", "wow", "perfect", "sad", "end", "awesome", "happy", "beautiful", "fantastic", "definitely", "heart", "finished", "highly", "incredible", "brilliant", "god", "truly", "writing", "cried", "simply", "right", "satisfying", "conclusion", "holy", "epic", "sequel", "coming", "favorite", "page", "believe", "written", "wish", "thank", "left", "excited", "begging", "tears", "alert", "beautifully", "plot", "hope", "completely", "way", "fantasy", "life", "adore", "fabulous", "perfectly", "suspense", "ready", "breathtaking", "mind-blowing", "stunning", "fast", "funny", "awesome", "phenomenal", "captivating", "heartfelt", "imagination", "unique", "perfection", "amazing", "adoring", "great", "incredible"],
}

subgenres = list(subgenre_dict.keys())
flattened_subgenre_words = [' '.join(words) for words in subgenre_dict.values()]

# Create TF-IDF vectors for top 10 words of each topic and subgenre words
print("Calculating TF-IDF vectors for top words...")
combined_words = [' '.join(words) for words in top_10_words.values()] + flattened_subgenre_words
tfidf_matrix_combined = vectorizer.transform(combined_words)

# Split the TF-IDF matrix back into topic and subgenre vectors
topic_vectors = tfidf_matrix_combined[:len(top_10_words)]
subgenre_vectors = tfidf_matrix_combined[len(top_10_words):]

# Calculate cosine similarity between topic and subgenre vectors
print("Calculating cosine similarities...")
cosine_similarities = cosine_similarity(topic_vectors, subgenre_vectors)

# Create a cosine similarity matrix
cosine_similarity_matrix = pd.DataFrame(cosine_similarities, index=[f"Topic {i}" for i in range(len(top_10_words))], columns=subgenres)

# Visualize the similarities using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_similarity_matrix, annot=True, cmap="coolwarm")
plt.title("Cosine Similarity between NMF Topics and Subgenres")
plt.show()

# Save the cosine similarity matrix
cosine_matrix_path = os.path.join(output_dir, "cosine_similarity_matrix_nmf.csv")
cosine_similarity_matrix.to_csv(cosine_matrix_path)

# Assign subgenre labels to each document
print("Assigning labels...")
nmf_topic_matrix = nmf_model.transform(tfidf_matrix)
dominant_topics = np.argmax(nmf_topic_matrix, axis=1)

# Map each NMF topic to the most similar subgenre (this part needs actual subgenres)
# topic_to_subgenre = cosine_similarity_matrix.idxmax(axis=1).to_dict()

# Placeholder for subgenre assignment, replace with actual labels
subgenre_labels = [f"Subgenre {topic}" for topic in dominant_topics]

# Add the new column to the DataFrame
df['nmf_label'] = subgenre_labels

# Save the DataFrame with NMF labels
nmf_labels_path = os.path.join(output_dir, "nmf_labels.feather")
df[['nmf_label']].to_feather(nmf_labels_path)

print("Labels assigned and DataFrame saved.")
