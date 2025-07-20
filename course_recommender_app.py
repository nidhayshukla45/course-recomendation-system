import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import difflib

# Load the CSV file
df = pd.read_csv("courses.csv")

# Page title
st.title("🎓 Course Recommendation System")

# Text input
user_input = st.text_input("Enter your skill or interest:")

# When button is clicked
if st.button("Recommend Course"):
    if user_input.strip() == "":
        st.warning("Please enter something.")
    else:
        # Lowercase matching
        user_input_cleaned = user_input.strip().lower()
        possible_matches = difflib.get_close_matches(user_input_cleaned, df['skill'].str.lower(), n=1, cutoff=0.6)

        if possible_matches:
            corrected_input = possible_matches[0]

            # Build and train the model
            cv = CountVectorizer()
            X = cv.fit_transform(df['skill'])
            y = df['course']
            model = MultinomialNB()
            model.fit(X, y)

            # Predict using corrected input
            user_vector = cv.transform([corrected_input])
            prediction = model.predict(user_vector)

            # Show result
            if corrected_input != user_input_cleaned:
                st.success(f"✅ Recommended Course: {prediction[0]} (matched with: {corrected_input})")
            else:
                st.success(f"✅ Recommended Course: {prediction[0]}")
        else:
            st.error("❌ No match found. Try something like 'python', 'data science', etc.")

