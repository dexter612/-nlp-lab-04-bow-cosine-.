import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# NLP LAB EXERCISE 04
# Topic: Vector Space Modeling - Bag of Words & Cosine Similarity
# ============================================================

# -------------------------
# TASK 1: Bag of Words
# -------------------------

corpus = [
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]

# Create CountVectorizer with English stop words removed
vectorizer = CountVectorizer(stop_words="english")

# Convert corpus into Bag of Words matrix
bow_matrix = vectorizer.fit_transform(corpus)

# Extract vocabulary
vocabulary = vectorizer.get_feature_names_out()

# Convert matrix into a Pandas DataFrame
bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vocabulary,
    index=["Review 1", "Review 2", "Review 3"]
)

print("=" * 60)
print("TASK 1: BAG OF WORDS MATRIX")
print("=" * 60)

print("\nVocabulary:")
print(list(vocabulary))

print("\nBag of Words Matrix:")
print(bow_df)


# -------------------------
# TASK 2: Document Search Engine
# -------------------------

documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

query = ["machine learning algorithms for data"]

# Fit vectorizer on documents
search_vectorizer = CountVectorizer()

# Convert documents and query into numerical vectors
doc_vectors = search_vectorizer.fit_transform(documents)
query_vector = search_vectorizer.transform(query)

# Calculate cosine similarity between query and every document
similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]

# Create ranking DataFrame
ranking_df = pd.DataFrame({
    "Document": [f"Document {i}" for i in range(1, len(documents) + 1)],
    "Cosine Similarity": similarity_scores
})

# Sort from highest score to lowest score
ranking_df = ranking_df.sort_values(
    by="Cosine Similarity",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("TASK 2: DOCUMENT SEARCH ENGINE & RELEVANCE RANKING")
print("=" * 60)

print("\nRanked Documents:")
for rank, row in ranking_df.iterrows():
    doc_number = int(row["Document"].split()[1])
    print(
        f"Rank {rank + 1}: Document {doc_number} "
        f"- Cosine Similarity = {row['Cosine Similarity']:.4f}"
    )
    print(f"  Text: {documents[doc_number - 1]}")

print("\n" + "=" * 60)
print("=" * 60)
