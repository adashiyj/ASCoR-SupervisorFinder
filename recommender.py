import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import _stop_words

# Load precomputed data (vectorizer, TF-IDF matrix, researcher info)
with gzip.open("precomputed.pkl.gz", "rb") as f:
    data = pickle.load(f)

researcher_works = data["researcher_works"]
researcher_names = list(researcher_works.keys())
tfidf_matrix = data["tfidf_matrix"]
vectorizer = data["vectorizer"]
researcher_top_keywords = data["researcher_top_keywords"]

# Simple preprocessing function (no spaCy)
def preprocess(text):
    tokens = [t.lower() for t in text.split() if t.isalpha() and t.lower() not in _stop_words.ENGLISH_STOP_WORDS]
    return " ".join(tokens)

def recommend_supervisors(user_input, top_n=3, top_papers=3):
    user_text = preprocess(user_input)
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        researcher = researcher_names[idx]
        works = researcher_works[researcher]

        paper_texts = []
        paper_dois = []
        for w in works:
            text = (w["title"] + " " + w.get("abstract", "")).lower()
            paper_texts.append(text)
            paper_dois.append(w.get("doi", ""))

        paper_vecs = vectorizer.transform(paper_texts)
        paper_sims = cosine_similarity(user_vec, paper_vecs).flatten()
        top_paper_indices = paper_sims.argsort()[::-1][:top_papers]

        top_paper_list = [{"doi": paper_dois[i], "similarity": float(paper_sims[i])} for i in top_paper_indices]

        recommendations.append({
            "researcher": researcher,
            "top_keywords": researcher_top_keywords.get(researcher, []),
            "similarity_score": float(similarities[idx]),
            "top_papers": top_paper_list
        })

    return recommendations
