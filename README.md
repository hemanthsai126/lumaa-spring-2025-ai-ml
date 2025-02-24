# A Presentation for Luma AI

**Open this link to test the app:**  
[https://movie-recommendation-system-hemanthsai.streamlit.app](https://movie-recommendation-system-hemanthsai.streamlit.app)

## Overview

This repository demonstrates a **content-based recommendation system** for movies using **manual TF-IDF** and **cosine similarity**. Given a user’s short text description, the system returns the **top recommended movies** from a small dataset of 500 entries. This solution is prepared as a presentation for **Luma AI** to showcase how quickly and effectively such a system can be built.

## Files in This Repository

1. **`output.csv`**  
   - The dataset of 500 movies. Each row has:
     - **title**
     - **plot** (2–3 sentences, first sentence mentioning the genre)

2. **`main.py`**  
   - Contains the logic to:
     - **Load and preprocess** `output.csv`
     - **Build** TF-IDF vectors manually (no scikit-learn’s `TfidfVectorizer`)
     - **Compute** cosine similarity manually
     - **Recommend** top N movies for a given user query

3. **`app.py`**  
   - A **Streamlit** UI that:
     - **Loads** the resources from `main.py` (cached for efficiency)
     - **Takes** user input via a text box
     - **Displays** the top 5 recommended movies, with a **Lottie animation** for style

## Setup

**Python Version**: 3.7+ recommended

### Optional Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# or venv\Scripts\activate      # Windows

