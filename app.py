import streamlit as st
import requests
from streamlit_lottie import st_lottie
import main 


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.set_page_config(
    page_title="Movie Recommender",
    page_icon=":clapper:",
    layout="centered"
)

st.title("Movie Recommendation System :clapper:")
st.markdown(
    "Welcome to the **Movie Recommender**! "
    "Just describe what kind of movie you're in the mood for, "
    "and we’ll suggest the top matches."
)

lottie_url = "https://assets1.lottiefiles.com/packages/lf20_jwalah3c.json"  
lottie_json = load_lottieurl(lottie_url)
if lottie_json:
    st_lottie(lottie_json, height=250, key="lottie_animation")


@st.cache_data  # or st.cache (older versions of Streamlit)
def load_resources():
    df, vocab, idf_values, doc_tfidf_vectors = main.load_and_prepare_data("output.csv")
    return df, vocab, idf_values, doc_tfidf_vectors


df, vocab, idf_values, doc_tfidf_vectors = load_resources()

user_query = st.text_input("What kind of movie are you looking for today?")

if st.button("Recommend"):
    if not user_query.strip():
        st.warning("Please enter a description of your movie preference.")
    else:
        
        top_recs = main.recommend_movies(
            user_query=user_query,
            df=df,
            vocab=vocab,
            idf_values=idf_values,
            doc_tfidf_vectors=doc_tfidf_vectors,
            top_n=5
        )

        
        st.subheader("Top Recommendations")
        for idx, row in top_recs.iterrows():
            st.markdown(f"**{row['title']}**  \nSimilarity: {row['similarity']:.4f}")
