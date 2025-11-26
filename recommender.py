import json
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy and add stopwords
nlp = spacy.load("en_core_web_sm")
custom_stopwords = {"study", "studies", "result", "results", "analysis", "research", "find", "found", "seim", "change", "de", "en", "e", "em", "o"}
for w in custom_stopwords:
    nlp.vocab[w].is_stop = True

# Load researcher works
with open("data/researcher_works.json", "r", encoding="utf-8") as fi:
    researcher_works = json.load(fi)

def build_abstract(index):
    if not index:
        return ""
    words = sorted(
        [(pos, word) for word, positions in index.items() for pos in positions],
        key=lambda x: x[0]
    )
    return " ".join([w for _, w in words])

# Create dicts for corpus and top keywords
researcher_top_keywords = {}
researcher_corpus = {}

for researcher, works in researcher_works.items():
    word_counter = Counter()
    corpus_text = []

    for w in works:
        abstract = build_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        doc = nlp(abstract.lower())
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks if not any(token.is_stop for token in chunk)]
        word_counter.update(tokens + phrases)
        corpus_text.append(" ".join(tokens + phrases))

    researcher_top_keywords[researcher] = [kw for kw, _ in word_counter.most_common(10)]
    researcher_corpus[researcher] = " ".join(corpus_text)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(list(researcher_corpus.values()))
researcher_names = list(researcher_corpus.keys())

def recommend_supervisors(user_input, top_n=3, top_papers=3):
    doc = nlp(user_input.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    phrases = [chunk.text.lower() for chunk in doc.noun_chunks if not any(token.is_stop for token in chunk)]
    user_text = " ".join(tokens + phrases)
    
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        researcher = researcher_names[idx]
        works = researcher_works[researcher]
        paper_texts, paper_dois = [], []

        for w in works:
            abstract = build_abstract(w.get("abstract_inverted_index"))
            text = (w.get("title", "") + " " + abstract).lower()
            paper_texts.append(text)
            paper_dois.append(w.get("doi", "No DOI"))

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