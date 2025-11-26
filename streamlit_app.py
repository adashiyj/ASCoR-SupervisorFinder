import streamlit as st
from recommender import recommend_supervisors

st.set_page_config(page_title="ASCoR Master's Thesis Supervisor Finder", page_icon="🎓")

st.title("ASCoR Master's Thesis Supervisor Finder")
st.write(
    "Tell me something about your research interests and what you would like to do for your thesis."
)

# User input
user_input = st.text_area("Your research interests", "", height=150)


if st.button("Find Supervisors") and user_input.strip():
    with st.spinner("Finding the best supervisors..."):
        results = recommend_supervisors(user_input)

    if results:
        for res in results:
            st.subheader(res["researcher"])
            st.write(f"**Similarity Score:** {res['similarity_score']:.3f}")
            st.write("**Top Keywords:**", ", ".join(res["top_keywords"]))
            st.write("**Top Papers:**")
            for paper in res["top_papers"]:
                st.markdown(f"- [{paper['doi']}]({paper['doi']}) – Similarity: {paper['similarity']:.3f}")
    else:
        st.write("Try refining your input.")
