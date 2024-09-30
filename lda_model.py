# LDA Model for Topic Modeling
# Define your directories
dir = "/path/to/your/dataset"
model_dir = "/path/to/your/models"
output_dir = "/path/to/your/output"

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your cleaned data
print("Loading cleaned data...")
df = pd.read_feather(os.path.join(dir, "processed_reviews.feather"))

# Preprocess the text
print("Preprocessing text...")
texts = df['cleaned_text'].tolist()

# Create a dictionary and corpus
print("Creating dictionary and corpus...")
dictionary = corpora.Dictionary([text.split() for text in texts])
corpus = [dictionary.doc2bow(text.split()) for text in texts]

# Print some statistics
print(f"Number of documents: {len(texts)}")
print(f"Number of unique tokens: {len(dictionary)}")
print(f"Number of documents (corpus): {len(corpus)}")

# Create the LDA model
num_topics = 10
print(f"Creating LDA model with {num_topics} topics...")
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, workers=4)

# Save the model
model_path = os.path.join(model_dir, "lda_model.model")
print(f"Saving model to {model_path}...")
lda_model.save(model_path)

# Load the saved model
print(f"Loading model from {model_path}...")
loaded_lda_model = LdaMulticore.load(model_path)

# Print the topics to verify the model
print("LDA topics from loaded model:")
for idx, topic in loaded_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

print("LDA process completed.")

# Extract top 200 words from each topic
num_words = 200
topics = lda_model.show_topics(num_topics=10, num_words=num_words, formatted=False)
topic_words = {i: [word for word, _ in topic] for i, topic in topics}

# Convert topic words to a DataFrame
topic_words_df = pd.DataFrame(topic_words)

# Save topic words to a CSV for review
topic_words_csv = os.path.join(output_dir, "topic_words.csv")
topic_words_df.to_csv(topic_words_csv, index=False)

# Subgenre dictionary (should be predefined)
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

# Assuming `subgenre_dict` and `topic_words` are already defined
subgenres = list(subgenre_dict.keys())
flattened_subgenre_words = [' '.join(words) for words in subgenre_dict.values()]

# Create TF-IDF vectors for topic words and subgenre words
vectorizer = TfidfVectorizer()
combined_words = [' '.join(words) for words in topic_words.values()] + flattened_subgenre_words
tfidf_matrix = vectorizer.fit_transform(combined_words)

# Split the TF-IDF matrix back into topic and subgenre vectors
topic_vectors = tfidf_matrix[:10]
subgenre_vectors = tfidf_matrix[10:]

# Calculate cosine similarity
cosine_similarities = cosine_similarity(topic_vectors, subgenre_vectors)

# Create a cosine similarity matrix
cosine_similarity_matrix = pd.DataFrame(cosine_similarities, index=[f"Topic {i}" for i in range(len(topic_words))], columns=subgenres)

# Visualize the similarities using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_similarity_matrix, annot=True, cmap="coolwarm")
plt.title("Cosine Similarity between LDA Topics and Subgenres")
plt.show()

# Save the cosine similarity matrix
cosine_matrix_path = os.path.join(output_dir, "cosine_similarity_matrix_lda.csv")
cosine_similarity_matrix.to_csv(cosine_matrix_path)

# Find the subgenre with the highest similarity for each topic
topic_to_subgenre = cosine_similarity_matrix.idxmax(axis=1).to_dict()

# Assign subgenre labels to each document
print("Assigning labels...")
subgenre_labels = []
for bow in corpus:
    topic_distribution = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
    subgenre_labels.append(topic_to_subgenre[f"Topic {dominant_topic}"])

# Add the new column to the DataFrame
df['lda_label'] = subgenre_labels

# Save the DataFrame as a Feather file
lda_labels_path = os.path.join(output_dir, "lda_labels.feather")
df[['lda_label']].to_feather(lda_labels_path)

print("Labels assigned and DataFrame saved.")
