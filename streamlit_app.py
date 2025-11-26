import streamlit as st
import gzip, pickle, spacy
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ASCoR Supervisor Finder")

st.title("🎓 ASCoR Thesis Supervisor Recommender")

st.write("""
Tell us your **research interests and early thesis ideas**.  
We'll recommend suitable ASCoR supervisors and link you to the most relevant DOIs from their work.
""")

# Load spaCy model
@st.cache_resource
nlp = spacy.load("en_core_web_sm") 

# Load precomputed recommender package
@st.cache_data
def load_package():
    try:
        with gzip.open("researcher_works.pkl.gz", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("Failed to load model.")
        st.code(str(e))
        return None

package = load_package()

if not package:
    st.stop()

vectorizer = package["vectorizer"]
tfidf_matrix = package["tfidf_matrix"]
researcher_corpus = package["researcher_corpus"]
researcher_top_keywords = package["researcher_top_keywords"]
researcher_works = package["researcher_works"]

researcher_names = list(researcher_corpus.keys())

user_input = st.text_area(
    "✍️ Describe your topic, approach, and possible methods:",
    placeholder="E.g. I study social robots and human-machine communication using surveys..."
)

if st.button("🔍 Recommend Supervisors"):
    if not user_input.strip():
        st.warning("Please enter your research interests!")
    else:
        doc = nlp(user_input.lower())
        tokens = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
        user_text = " ".join(tokens)

        user_vec = vectorizer.transform([user_text])
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:3]

        for i in top_idx:
            rname = researcher_names[i]
            st.subheader(rname)
            st.write(f"**Match score:** `{sims[i]:.3f}`")
            st.write("**Top expertise keywords:**", ", ".join(researcher_top_keywords.get(rname, [])))

            works = researcher_works.get(rname, [])
            if not works:
                st.write("_No publications found_")
            else:
                top_papers = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)[:3]
                st.write("**Relevant publications (DOIs):**")
                for p in top_papers:
                    st.write("-", p.get("doi", "No DOI"), f"(citations: {p.get('cited_by_count', 0)})")

            st.divider()



