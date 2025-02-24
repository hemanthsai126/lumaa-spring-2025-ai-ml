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







Install Dependencies
pip install -r requirements.txt
In requirements.txt, you should include:

streamlit
requests
pandas
numpy
streamlit-lottie


**How to Run Locally**

Copy or clone this repository so you have app.py, main.py, and output.csv in the same folder.
Open a terminal in that folder, optionally activate your virtual environment, then run:
streamlit run app.py
Streamlit will provide a local URL (usually http://localhost:8501). Open that in your browser.
Type a query (for example, "I love action movies set in space, with a comedic twist") and click Recommend.
The top 5 matches, along with similarity scores, will be displayed.
Example Usage (Streamlit)

From Terminal:
streamlit run app.py
In your browser, open the provided local URL.
Enter query:
"I love action movies set in space, with a comedic twist."
Click Recommend.
The recommended titles appear with their similarity scores.
Sample Output

User Query: I love action movies set in space, with a comedic twist.
Top 5 Recommendations:
 - Guardians of the Galaxy (similarity: 0.8623)
 - Thor: Ragnarok (similarity: 0.8150)
 - Star Wars: Episode IV (similarity: 0.7982)
 ...
Salary Expectation

My expected salary: $30 per hour

This project demonstrates a simple yet effective content-based recommender. You can easily enhance it with additional text processing or advanced weighting. The included dataset (output.csv) is kept small (500 rows) for quick demonstration.

