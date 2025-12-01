import pickle
import gzip
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed data
with gzip.open("precomputed.pkl.gz", "rb") as f:
    data = pickle.load(f)

researcher_works = data["researcher_works"]
researcher_names = list(researcher_works.keys())
tfidf_matrix = data["tfidf_matrix"]
vectorizer = data["vectorizer"]
researcher_interest_summary = data["researcher_interest_summary"]

def preprocess(text):
    """Simple preprocessing: lowercase, remove punctuation & stopwords."""
    return " ".join([t.lower() for t in text.split() if t.isalpha()])

def recommend_supervisors(user_input):
    user_text = preprocess(user_input)
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_n = 3        # top researchers
    top_papers = 3   # top papers per researcher

    top_indices = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        researcher = researcher_names[idx]
        works = researcher_works[researcher]

        paper_texts = []
        paper_dois = []
        for w in works:
            doi = w.get("doi")
            if not doi:  # Skip papers without DOI
                continue
            title = w.get("title") or ""
            abstract = w.get("abstract") or ""
            text = (title + " " + abstract).lower().strip()
            paper_texts.append(text)
            paper_dois.append(doi)

        if paper_texts:
            paper_vecs = vectorizer.transform(paper_texts)
            paper_sims = cosine_similarity(user_vec, paper_vecs).flatten()
            top_paper_indices = paper_sims.argsort()[::-1][:top_papers]

            top_paper_list = [
                {"doi": paper_dois[i], "similarity": float(paper_sims[i])}
                for i in top_paper_indices
            ]
        else:
            top_paper_list = []

        recommendations.append({
            "researcher": researcher,
            "researcher_interest_summary": researcher_interest_summary.get(researcher, ""),
            "similarity_score": float(similarities[idx]),
            "top_papers": top_paper_list
        })

    return recommendations
