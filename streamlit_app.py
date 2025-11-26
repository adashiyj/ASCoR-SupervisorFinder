import streamlit as st
import gzip
import pickle
import recommender

# Title
st.title("ASCoR Supervisor Finder")
st.write("Find suitable supervisors based on your research interests.")


# Input from user
user_input = st.text_area(
    "Tell us something about your research interests and what you would like to do for your thesis:",
    height=150
)

# Number of recommendations
num_results = st.slider("Number of supervisors to recommend:", 1, 10, 5)

# Recommend button
if st.button("Recommend Supervisors"):
    if not user_input.strip():
        st.warning("Please enter some text to get recommendations.")
    else:
        with st.spinner("Finding best supervisors..."):
            results = recommend_supervisors(user_input, researcher_data, top_n=num_results)
            
        if results:
            st.success(f"Top {len(results)} supervisors:")
            for i, res in enumerate(results, start=1):
                st.write(f"**{i}. {res['name']}**")
                st.write(f"   Affiliation: {res.get('affiliation', 'N/A')}")
                st.write(f"   Expertise: {res.get('expertise', 'N/A')}")
                st.write("---")
        else:
            st.info("No suitable supervisors found.")
