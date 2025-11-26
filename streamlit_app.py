import streamlit as st
from recommender import recommend_supervisors

# Page configuration
st.set_page_config(
    page_title="ASCoR Thesis Supervisor Finder",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header
st.markdown(
    "<h1 style='display: inline; font-size: 2rem;'> ASCoR Master's Thesis Supervisor Finder ✨</h1>",
    unsafe_allow_html=True
)

# Add some vertical space
st.markdown("<br>", unsafe_allow_html=True)

# Description
st.markdown(
    """
    Welcome! This is a **content-based recommender system** designed to help 
    Computer Science students find a suitable thesis supervisor based on their research interests.
    Please describe your research interests and what you would like to do for your thesis below.
    """,
)

# User input
user_input = st.text_area(
    "📝 Enter your research interests and thesis ideas:",
    placeholder="e.g., I am interested in political communication, especially...",
    height=180
)

# Find supervisors button
# Find supervisors button
if st.button("🔍 Find My Supervisor(s)"):
    if user_input.strip():
        with st.spinner("Searching for the best supervisors..."):
            results = recommend_supervisors(user_input)

        if results:
            st.success(f"Found {len(results)} supervisor(s) matching your research interests!")
            for res in results:
                st.markdown(f"### {res['researcher']}")
                st.markdown(f"**Similarity Score:** {res['similarity_score']:.3f}")
                st.markdown(f"**Top Keywords:** {', '.join(res['top_keywords'])}")
                st.markdown("**Top Papers:**")
                for paper in res["top_papers"]:
                    st.markdown(
                        f"- [{paper['doi']}]({paper['doi']}) – Similarity: {paper['similarity']:.3f}"
                    )
                st.markdown("---")
        else:
            st.warning("No suitable supervisors found. Please refine your input and try again.")
    else:
        st.info("Please enter your research interests above before searching.")


