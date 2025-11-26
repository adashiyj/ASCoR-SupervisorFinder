import gzip
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed data
with gzip.open("precomputed.pkl.gz", "rb") as f:
    data = pickle.load(f)

researcher_works = data["researcher_works"]
researcher_corpus = data["researcher_corpus"]
researcher_top_keywords = data["researcher_top_keywords"]
vectorizer = data["vectorizer"]
tfidf_matrix = data["tfidf_matrix"]
researcher_names = list(researcher_corpus.keys())

def recommend_supervisors(user_input, top_n=3, top_papers=3):
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(user_input.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    user_text = " ".join(tokens)

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
            text = (w["title"] + " " + w["abstract"]).lower()
            paper_texts.append(text)
            paper_dois.append(w["doi"])

        paper_vecs = vectorizer.transform(paper_texts)
        paper_sims = cosine_similarity(user_vec, paper_vecs).flatten()
        top_paper_indices = paper_sims.argsort()[::-1][:top_papers]

        top_paper_list = [{"doi": paper_dois[i], "similarity": paper_sims[i]} for i in top_paper_indices]

        recommendations.append({
            "researcher": researcher,
            "top_keywords": researcher_top_keywords[researcher],
            "similarity_score": similarities[idx],
            "top_papers": top_paper_list
        })

    return recommendations
