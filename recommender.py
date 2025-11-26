import pickle
import gzip
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Add custom stopwords
custom_stopwords = {"study", "studies", "result", "results", "analysis",
                    "research", "find", "found", "seim", "change", "de", "en", "e", "em", "o"}
for w in custom_stopwords:
    nlp.vocab[w].is_stop = True

# Load compressed data file
with gzip.open("researcher_works.pkl.gz", "rb") as f:
    researcher_works = pickle.load(f)

def build_abstract(index):
    """Reconstruct abstract from OpenAlex inverted index."""
    if not index:
        return ""
    words = sorted(
        [(pos, word) for word, positions in index.items() for pos in positions],
        key=lambda x: x[0]
    )
    return " ".join([w for _, w in words])

# Build corpus and top keywords
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

        # Unigrams
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        word_counter.update(tokens)

        # Noun chunks (multi-word expressions)
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks
                   if not any(token.is_stop for token in chunk)]
        word_counter.update(phrases)

        # For TF-IDF corpus
        corpus_text.append(" ".join(tokens + phrases))

    # Top 10 keywords per researcher
    top_keywords = [kw for kw, _ in word_counter.most_common(10)]
    researcher_top_keywords[researcher] = top_keywords

    # Full text for TF-IDF matching
    researcher_corpus[researcher] = " ".join(corpus_text)

# Build TF-IDF matrix (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(list(researcher_corpus.values()))
researcher_names = list(researcher_corpus.keys())


def recommend_supervisors(user_input, top_n=3, top_papers=3):
    # Preprocess user input
    doc = nlp(user_input.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    phrases = [chunk.text.lower() for chunk in doc.noun_chunks if not any(token.is_stop for token in chunk)]
    user_text = " ".join(tokens + phrases)

    # Vectorize user input
    user_vec = vectorizer.transform([user_text])

    # Similarity to each researcher
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        researcher = researcher_names[idx]
        works = researcher_works[researcher]

        # Compute similarity for each paper
        paper_texts = []
        paper_dois = []
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
