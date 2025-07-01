# UI Setup

This document explains how to run the optional Streamlit interface for querying memory.

## Prerequisites

- Python 3.9 or later
- A running backend that exposes the `/memory/query` endpoint

## Installation

1. Create a virtual environment (optional but recommended).
2. Install the required packages:

   ```bash
   pip install -r ui/requirements.txt
   ```

## Running the App

1. Ensure the backend service is running. By default the UI sends requests to `http://localhost:8000`. You can override this by setting the `MEMORY_API_URL` environment variable.
2. Start the Streamlit app:

   ```bash
   streamlit run ui/app.py
   ```

3. Open the provided URL in your browser and query memory facts.

