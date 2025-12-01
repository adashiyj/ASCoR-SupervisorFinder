import pickle
import gzip
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import _stop_words
from keybert import KeyBERT

# Load the model
keybert_model = KeyBERT()

# Load precomputed data (vectorizer, TF-IDF matrix, researcher info)
with gzip.open("precomputed.pkl.gz", "rb") as f:
    data = pickle.load(f)

researcher_works = data["researcher_works"]          # original paper info
researcher_names = list(researcher_works.keys())
researcher_embeddings = data["researcher_embeddings"]  # dict: researcher -> paper embeddings (numpy arrays)
researcher_top_keywords = data.get("researcher_top_keywords", {}) 

# Simple preprocessing function (no spaCy)
def preprocess(text):
    tokens = [t.lower() for t in text.split() if t.isalpha() and t.lower() not in _stop_words.ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# Recommender function
def recommend_supervisors(user_input):
    """
    Recommend 3 researchers and their 3 most relevant papers with DOI.
    """
    top_n = 3        # top researchers
    top_papers = 3   # top papers per researcher

    # 1. Embed user input
    user_text = preprocess(user_input)
    user_emb = keybert_model._model.encode([user_text])[0].reshape(1, -1)

    # 2. Compute similarity with researcher papers
    researcher_scores = []
    for researcher in researcher_names:
        papers_emb = researcher_embeddings[researcher]
        sim_scores = cosine_similarity(user_emb, papers_emb).flatten()
        avg_sim = sim_scores.mean()
        researcher_scores.append((researcher, avg_sim, sim_scores))

    # 3. Select top 3 researchers
    top_researchers = sorted(researcher_scores, key=lambda x: x[1], reverse=True)[:top_n]

    # 4. Build recommendation output
    recommendations = []
    for researcher, avg_sim, paper_sims in top_researchers:
        works = researcher_works[researcher]
        # Keep only papers with DOI
        paper_info = [
            {"doi": w.get("doi"), "similarity": float(sim)}
            for w, sim in zip(works, paper_sims)
            if w.get("doi")
        ]
        # Sort and select top 3 papers
        paper_info = sorted(paper_info, key=lambda x: x["similarity"], reverse=True)[:top_papers]

        recommendations.append({
            "researcher": researcher,
            "top_keywords": researcher_top_keywords.get(researcher, []),
            "similarity_score": float(avg_sim),
            "top_papers": paper_info
        })

    return recommendations
