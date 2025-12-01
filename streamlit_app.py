import streamlit as st
import gzip
import pickle
from recommender import recommend_supervisors

# ---- Load precomputed data ----
@st.cache_data
def load_data(pkl_file="precomputed_data.pkl.gz"):
    with gzip.open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

# Load precomputed TF-IDF, researcher_corpus, researcher_top_keywords, researcher_works, vectorizer, researcher_names
precomputed = load_data()
vectorizer = precomputed["vectorizer"]
tfidf_matrix = precomputed["tfidf_matrix"]
researcher_names = precomputed["researcher_names"]
researcher_top_keywords = precomputed["researcher_top_keywords"]
researcher_works = precomputed["researcher_works"]

st.set_page_config(page_title="ASCoR Supervisor Finder", layout="wide")

st.title("ASCoR Supervisor Finder")
st.markdown(
    "Write down your research interests and preliminary ideas for your thesis. "
    "The system will suggest suitable supervisors along with their top publications."
)

# ---- User input ----
user_input = st.text_area(
    "Enter your research interests here:", 
    height=150
)

top_n = st.slider("Number of supervisors to recommend:", 1, 5, 3)
top_papers = st.slider("Number of top papers per supervisor:", 1, 5, 3)

if st.button("Find Supervisors"):
    if not user_input.strip():
        st.warning("Please enter your research interests first!")
    else:
        recommendations = recommend_supervisors(
            user_input, 
            top_n=top_n, 
            top_papers=top_papers,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            researcher_names=researcher_names,
            researcher_top_keywords=researcher_top_keywords,
            researcher_works=researcher_works
        )

        for r in recommendations:
            st.subheader(r["researcher"])
            st.markdown(f"**Top keywords:** {', '.join(r['top_keywords'])}")
            st.markdown(f"**Similarity score:** {r['similarity_score']:.3f}")
            
            if r["top_papers"]:
                st.markdown("**Top papers:**")
                for paper in r["top_papers"]:
                    st.markdown(f"- [{paper['doi']}]({paper['doi']}) (similarity: {paper['similarity']:.3f})")
            st.markdown("---")


