# main.py

import math
import re
import pandas as pd

CUSTOM_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "you",
    "your", "about", "they", "their", "have", "has", "will", "would",
    "could", "should", "make", "them", "those", "been", "were",
    "just", "really","like", "when", "where","is", "are","was","were"
    "here", "there", "into", "only", "says", "over", "some"
}

GENRE_KEYWORDS = {
    "action", "comedy", "drama", "romance", "horror", "thriller",
    "fantasy", "animation", "western", "space", "sci-fi", "science-fiction",
    "superhero", "crime", "biographical", "historical", "music",
    "musical", "adventure", "family", "sports", "teen", "love"
}


def extract_genres(text: str):
    if not isinstance(text, str):
        return set()
    sentences = text.split('.')
    first_sentence = sentences[0].lower() if sentences else ""

    found_genres = set()
    for genre in GENRE_KEYWORDS:
        if genre in first_sentence:
            found_genres.add(genre)
    return found_genres


def tokenize_and_clean(text: str):
    if not isinstance(text, str):
        return []

    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)  # keep only alphabetical
    cleaned_tokens = [
        t for t in tokens
        if len(t) >= 4 and t not in CUSTOM_STOPWORDS
    ]
    return cleaned_tokens


def build_vocab_and_df(docs):
    vocab = {}
    doc_freq = {}
    current_index = 0

    for tokens in docs:
        unique_words = set(tokens)
        # doc frequency
        for w in unique_words:
            doc_freq[w] = doc_freq.get(w, 0) + 1
        # build vocab
        for w in tokens:
            if w not in vocab:
                vocab[w] = current_index
                current_index += 1
    return vocab, doc_freq


def compute_idf(doc_freq, total_docs: int):
    idf_values = {}
    for word, freq in doc_freq.items():
        idf_values[word] = math.log(total_docs / (1.0 + freq))
    return idf_values


def build_tfidf_matrix(tokenized_docs, vocab, idf_values):
    tfidf_matrix = []
    vocab_size = len(vocab)

    for tokens in tokenized_docs:
        counts = {}
        for w in tokens:
            counts[w] = counts.get(w, 0) + 1

        doc_len = len(tokens)
        vector = [0.0] * vocab_size

        for w, c in counts.items():
            if w in vocab:  # word is known
                tf = c / doc_len
                tfidf = tf * idf_values[w]
                idx = vocab[w]
                vector[idx] = tfidf

        tfidf_matrix.append(vector)

    return tfidf_matrix


def build_query_vector(tokens, vocab, idf_values):
    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1

    vector = [0.0] * len(vocab)
    length = len(tokens)

    for w, c in counts.items():
        if w in vocab and w in idf_values:
            tf = c / length
            tfidf = tf * idf_values[w]
            idx = vocab[w]
            vector[idx] = tfidf

    return vector


def compute_cosine_similarity(vec1, vec2):
    dot_prod = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_prod / (norm1 * norm2)

def load_and_prepare_data(csv_path="output.csv"):
    df = pd.read_csv(csv_path)
    df["genres"] = df["plot"].apply(extract_genres)

    tokenized_docs = [tokenize_and_clean(plot) for plot in df["plot"]]
    vocab, doc_freq = build_vocab_and_df(tokenized_docs)
    total_docs = len(tokenized_docs)
    idf_values = compute_idf(doc_freq, total_docs)

    doc_tfidf_vectors = build_tfidf_matrix(tokenized_docs, vocab, idf_values)

    return df, vocab, idf_values, doc_tfidf_vectors

def recommend_movies(user_query, df, vocab, idf_values, doc_tfidf_vectors, top_n=5):
    user_genres = extract_genres(user_query)
    user_tokens = tokenize_and_clean(user_query)
    user_vec = build_query_vector(user_tokens, vocab, idf_values)

    sims = []
    for i, doc_vec in enumerate(doc_tfidf_vectors):
        cos_sim = compute_cosine_similarity(user_vec, doc_vec)

        overlap = user_genres & df.loc[i, "genres"]
        if overlap:
            final_sim = 0.5 + cos_sim
        else:
            final_sim = cos_sim

        if final_sim > 1.0:
            final_sim = 1.0

        sims.append((final_sim, i))


    sims.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for (sim, idx) in sims[:top_n]]
    top_sims = [sim for (sim, idx) in sims[:top_n]]

    results = df.iloc[top_indices].copy()
    results["similarity"] = top_sims
    results = results.sort_values("similarity", ascending=False)
    return results
