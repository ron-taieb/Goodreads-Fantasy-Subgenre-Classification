# BERTopic Model for Topic Modeling
# Define your directories
dir = "/path/to/your/dataset"
model_dir = "/path/to/your/models"
output_dir = "/path/to/your/output"

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load your cleaned data
print("Loading cleaned data...")
df = pd.read_feather(os.path.join(dir, "processed_reviews.feather"))

# Initialize sub-models that support online learning
print("Initializing sub-models for online learning...")
embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
umap_model = PCA(n_components=5)
cluster_model = MiniBatchKMeans(n_clusters=10, random_state=0)
vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=0.01, min_df=10)

# Initialize the BERTopic model
print("Initializing BERTopic model...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    low_memory=True,
    min_topic_size=200,
    nr_topics="auto",
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    representation_model=MaximalMarginalRelevance(diversity=0.7)
)

# Fit the model to your data in batches
print("Fitting BERTopic model in batches...")
batch_size = 100000  # Adjust the batch size according to your memory constraints
texts = df['cleaned_text'].tolist()

total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)

for i in range(0, len(texts), batch_size):
    batch_num = i // batch_size + 1
    batch_texts = texts[i:i + batch_size]
    print(f"Processing batch {batch_num}/{total_batches}...")
    topic_model.partial_fit(batch_texts)

# Save the BERTopic model
print("Saving BERTopic model...")
with open(os.path.join(model_dir, "bertopic_model_online.pkl"), 'wb') as model_file:
    pickle.dump(topic_model, model_file)

# Extract top 200 words for each topic
print("Extracting top 200 words for each topic...")
topics = topic_model.get_topics()
topic_words = {f"Topic {topic_num}": [word for word, _ in topic_model.get_topic(topic_num)[:200]] for topic_num in topics.keys()}

# Subgenre dictionary (predefined)
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

# Create TF-IDF vectors for top 200 words of each topic and subgenre words
print("Calculating TF-IDF vectors for top words...")
vectorizer = CountVectorizer(stop_words="english")
combined_words = [' '.join(words) for words in topic_words.values()] + flattened_subgenre_words
tfidf_matrix = vectorizer.fit_transform(combined_words)

# Split the TF-IDF matrix back into topic and subgenre vectors
topic_vectors = tfidf_matrix[:len(topics)]
subgenre_vectors = tfidf_matrix[len(topics):]

# Calculate cosine similarity
print("Calculating cosine similarities...")
cosine_similarities = cosine_similarity(topic_vectors, subgenre_vectors)

# Create a cosine similarity matrix
cosine_similarity_matrix = pd.DataFrame(cosine_similarities, index=[f"Topic {i}" for i in range(len(topics))], columns=subgenres)

# Sort the matrix rows by the maximum value in each row
cosine_similarity_matrix['max_value'] = cosine_similarity_matrix.max(axis=1)
cosine_similarity_matrix = cosine_similarity_matrix.sort_values(by='max_value', ascending=False).drop(columns='max_value')

# Visualize the similarities using a heatmap with annotations
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_similarity_matrix, annot=True, cmap='coolwarm')
plt.title("Cosine Similarity between BERTopic Topics and Subgenres")
plt.xticks(rotation=45, ha='right')
plt.show()

# Save the cosine similarity matrix
cosine_matrix_path = os.path.join(output_dir, "cosine_similarity_matrix_bertopic.csv")
cosine_similarity_matrix.to_csv(cosine_matrix_path)

# Map each BERTopic topic to the most similar subgenre
topic_to_subgenre = cosine_similarity_matrix.idxmax(axis=1).to_dict()

# Assign subgenre labels to each document
print("Assigning labels...")
dominant_topics = topic_model.transform(texts)
subgenre_labels = [topic_to_subgenre[f"Topic {topic}"] for topic in dominant_topics]

# Add the new column to the DataFrame
df['bertopic_label'] = subgenre_labels

# Save the DataFrame as a Feather file
output_feather_path = os.path.join(output_dir, "bertopic_labels_online.feather")
df[['bertopic_label']].to_feather(output_feather_path)

print("Labels assigned and DataFrame saved.")
