import os
import requests
import streamlit as st

API_URL = os.getenv("MEMORY_API_URL", "http://localhost:8000")

st.title("Memory Query UI")

query = st.text_input("Enter query")

if st.button("Search"):
    if query:
        try:
            response = requests.post(f"{API_URL}/memory/query", json={"query": query})
            response.raise_for_status()
            data = response.json()
            st.subheader("Retrieved Facts")
            st.json(data)
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a query first.")

