import streamlit as st
from utils.recommender import recommend_supervisors

st.set_page_config(page_title="ASCoR Master's Thesis Supervisor Recommender", layout="wide")

st.title("ASCoR Master's Thesis Supervisor Recommender")
st.write(
    "Share your research interests and preliminary ideas for your thesis below. "
    "The system will suggest suitable supervisors at ASCoR and highlight some of their most relevant publications."
)

# User input
user_input = st.text_area(
    "Your research interests:",
    "I am interested in human machine communication. I want to study the interaction with social robots."
)

# Options
top_n = st.slider("Number of supervisors to recommend:", 1, 10, 3)
top_papers = st.slider("Number of top papers per supervisor:", 1, 5, 3)

if st.button("Recommend"):
    if not user_input.strip():
        st.warning("Please enter your research interests.")
    else:
        with st.spinner("Finding suitable supervisors..."):
            recommendations = recommend_supervisors(user_input, top_n=top_n, top_papers=top_papers)

        if not recommendations:
            st.info("No suitable supervisors found.")
        else:
            st.success(f"Top {len(recommendations)} supervisor(s) found!")

            for r in recommendations:
                # Main supervisor info in header
                similarity_pct = r['similarity_score'] * 100
                header = f"{r['researcher']} (Similarity: {similarity_pct:.1f}%)"
                
                # Expandable section for each supervisor
                with st.expander(header, expanded=False):
                    st.write(f"**Top keywords:** {', '.join(r['top_keywords'])}")
                    st.write("**Top papers:**")
                    for paper in r["top_papers"]:
                        sim_pct = paper["similarity"] * 100
                        st.write(f"- DOI: {paper['doi']} (Similarity: {sim_pct:.1f}%)")
