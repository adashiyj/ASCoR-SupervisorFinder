import streamlit as st
from recommender import recommend_supervisors

st.set_page_config(page_title="ASCoR Supervisor Finder", layout="wide")

st.title("ASCoR Supervisor Finder")
st.write(
    "Enter your research interests and preliminary thesis ideas below, "
    "and the system will suggest suitable supervisors along with their top publications (DOIs)."
)

user_input = st.text_area("Your research interests", height=150)

top_n = st.slider("Number of supervisors to recommend", 1, 5, 3)
top_papers = st.slider("Number of top publications per supervisor", 1, 5, 3)

if st.button("Recommend"):
    if user_input.strip():
        recommendations = recommend_supervisors(user_input, top_n=top_n, top_papers=top_papers)

        for r in recommendations:
            st.subheader(r["researcher"])
            st.write(f"Top keywords: {', '.join(r['top_keywords'])}")
            st.write(f"Similarity score: {r['similarity_score']:.3f}")
            st.write("Top papers:")
            for paper in r["top_papers"]:
                st.write(f" - DOI: {paper['doi']} (similarity: {paper['similarity']:.3f})")
            st.markdown("---")
    else:
        st.warning("Please enter your research interests before clicking Recommend.")

