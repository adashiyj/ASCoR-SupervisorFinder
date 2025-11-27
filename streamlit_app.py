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
    "<h1 style='display: inline; font-size: 2rem;'>🎓 Find My Thesis Supervisor at ASCoR ✨</h1>",
    unsafe_allow_html=True
)

# Add some vertical space
st.markdown("<br>", unsafe_allow_html=True)

# Description
st.markdown(
    """
    **Welcome!** 🎉  

    This is a **content-based recommender system** designed to help Communication Science students at the UvA
    find a suitable thesis supervisor based on their research interests and ideas.  

    After entering your research interests in the box below and clicking the **"Find My Supervisor(s)"** button, 
    the system will:  
    1. Analyze your input to identify key topics.  
    2. Compare your interests with available supervisors.  
    3. Provide a **ranked list of supervisors** that best match your interests.  
    4. Show each supervisor's **top keywords** in their publications and a few of their **most relevant papers** with links.  

    This way, you can quickly see which supervisors align with your research goals and explore their work.
    """
)

# Add some vertical space
st.markdown("<br>", unsafe_allow_html=True)

# User input
user_input = st.text_area(
    "📝 Enter your research interests and thesis ideas:",
    placeholder="e.g., I am interested in political communication, especially...",
    height=180
)

# Add some vertical space
st.markdown("<br>", unsafe_allow_html=True)

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
                st.markdown(f"**💡 Key Topics:** {', '.join(res['top_keywords'])}")
                st.markdown(f"** 📖 Relevant Papers:**")
                for paper in res["top_papers"]:
                    st.markdown(
                        f"- [{paper['doi']}]({paper['doi']}) – Similarity: {paper['similarity']:.3f}"
                    )
                st.markdown("---")
        else:
            st.warning("No suitable supervisors found. Please refine your input and try again.")
    else:
        st.info("Please enter your research interests above before searching.")


