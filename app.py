import os
import json
import pandas as pd
from serpapi import GoogleSearch
from urllib.parse import urlsplit, parse_qsl
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load API keys
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

def fetch_reviews(place_id, max_reviews=50):
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_maps_reviews",
        "hl": "en",
        "data_id": place_id
    }

    search = GoogleSearch(params)
    reviews = []
    while len(reviews) < max_reviews:
        results = search.get_dict()
        if "error" in results:
            print(results["error"])
            break
        for r in results.get("reviews", []):
            reviews.append({
                "Date": r.get("date", None), # Ensure 'Date' key always exists
                "Rating": r.get("rating"),
                "Review": r.get("snippet")
            })
            if len(reviews) >= max_reviews:
                break
        pagination = results.get("serpapi_pagination", {})
        if pagination.get("next") and pagination.get("next_page_token"):
            search.params_dict.update(dict(parse_qsl(urlsplit(pagination["next"]).query)))
        else:
            break
    return pd.DataFrame(reviews)

def enrich_reviews_batch(df, batch_size=10):
    """Enriches reviews using Gemini API in batches."""
    all_enriched_data = []
    df_reset = df.reset_index(drop=True) # Ensure default index for easy batch slicing

    for i in range(0, len(df_reset), batch_size):
        batch_df = df_reset[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}...")

        # Prepare batch data and prompt
        reviews_to_process = []
        original_indices = []
        for index, row in batch_df.iterrows():
             # Only process if review is valid text
             if isinstance(row.get("Review"), str) and row.get("Review"):
                 reviews_to_process.append(row["Review"])
                 original_indices.append(index) # Keep track of original position in batch_df
             else:
                 # Handle invalid/missing reviews within the batch immediately
                 all_enriched_data.append({
                     "Date": row.get("Date"), # Preserve original date
                     "Rating": row.get("Rating"),
                     "Review": row.get("Review"),
                     "Sentiment Score": pd.NA, # Indicate failure/skip
                     "Topics": "Invalid Review Data",
                     "Complaint?": pd.NA
                 })

        if not reviews_to_process: # Skip API call if batch has no valid reviews
             continue

        # Construct the prompt for the batch
        reviews_text = "\n---\n".join([f'Review {idx+1}:\n"{rev}"' for idx, rev in enumerate(reviews_to_process)])
        prompt = f"""Analyze the following {len(reviews_to_process)} reviews. For each review, provide a JSON object with keys: "SentimentScore" (0-10 integer), "Topics" (list of strings), and "Complaint" ("Yes" or "No"). Return a JSON list containing these objects, in the same order as the input reviews.

Reviews:
{reviews_text}

Respond ONLY with the JSON list. Example for 2 reviews:
[
  {{"SentimentScore": 8, "Topics": ["service", "food"], "Complaint": "No"}},
  {{"SentimentScore": 2, "Topics": ["wait time"], "Complaint": "Yes"}}
]
"""
        batch_results = [{} for _ in reviews_to_process] # Placeholder for results

        try:
            response = model.generate_content(prompt)
            response_text = getattr(response, 'text', '[]')
            # Clean the response text to extract the JSON list
            json_response_text = response_text.strip().strip('```json').strip('```').strip()
            if not json_response_text.startswith('['):
                 json_response_text = '[' + json_response_text # Attempt to fix if prefix missing
            if not json_response_text.endswith(']'):
                 json_response_text = json_response_text + ']' # Attempt to fix if suffix missing

            parsed_results = json.loads(json_response_text)

            if isinstance(parsed_results, list) and len(parsed_results) == len(reviews_to_process):
                batch_results = parsed_results
            else:
                print(f"Warning: Mismatch in expected ({len(reviews_to_process)}) vs received ({len(parsed_results)}) results for batch starting at index {i}.")
                # Keep batch_results as list of empty dicts, defaults will apply later

        except json.JSONDecodeError as json_e:
            print(f"Error decoding JSON response for batch starting at index {i}: {json_e}")
            print(f"Received text: {response_text[:500]}...") # Log received text
        except Exception as e:
            print(f"Error during Gemini API call for batch starting at index {i}: {e}")

        # Map results back to original batch rows
        result_idx = 0
        for original_batch_idx in original_indices:
             row_data = batch_df.loc[original_batch_idx]
             enrichment = batch_results[result_idx] if result_idx < len(batch_results) else {} # Get enrichment or empty dict

             all_enriched_data.append({
                 "Date": row_data.get("Date"), # Preserve original date
                 "Rating": row_data.get("Rating"),
                 "Review": row_data.get("Review"),
                 "Sentiment Score": enrichment.get("SentimentScore", pd.NA), # Use NA on failure
                 "Topics": ", ".join(enrichment.get("Topics", ["Enrichment Failed"])), # Indicate failure
                 "Complaint?": enrichment.get("Complaint", pd.NA) # Use NA on failure
             })
             result_idx += 1

    # Create final DataFrame
    columns_order = ["Date", "Rating", "Review", "Sentiment Score", "Topics", "Complaint?"]
    final_df = pd.DataFrame(all_enriched_data, columns=columns_order)

    # Attempt to convert Sentiment Score back to numeric, coercing errors
    final_df["Sentiment Score"] = pd.to_numeric(final_df["Sentiment Score"], errors='coerce')

    return final_df


def generate_summary(df):
    # Filter out rows where enrichment might have failed before summarizing
    valid_df = df.dropna(subset=['Sentiment Score', 'Complaint?'])
    if valid_df.empty:
        return "Could not generate summary as review enrichment failed for all entries."

    # Corrected prompt assignment (removed duplicate)
    prompt = f"""
    Analyze these {len(valid_df)} reviews (focusing on successfully analyzed ones):
    {valid_df[['Rating', 'Review']].to_string(index=False)}
    Summarize:
    1. Top 5 Themes mentioned in Topics
    2. Positive vs Negative Trends
    3. Any action items or issues?
    """
    response = model.generate_content(prompt)
    return response.text

def run_streamlit_app(df, summary):
    st.title("Google Review Insights Dashboard")

    st.subheader("Summary from Gemini")
    st.markdown(summary)

    st.subheader("Review Table")
    st.dataframe(df)

    st.subheader("Sentiment Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Sentiment Score"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Complaint Analysis")
    fig2, ax2 = plt.subplots()
    complaint_counts = df["Complaint?"].value_counts()
    sns.barplot(x=complaint_counts.index, y=complaint_counts.values, ax=ax2)
    ax2.set_ylabel("Number of Reviews")
    st.pyplot(fig2)

def main():
    place_id = st.text_input("Enter Google Maps Place ID:", "ChIJMbgb-JQjDTkREMhnXrrQk4w")
    if st.button("Fetch & Analyze Reviews"):
        with st.spinner("Fetching reviews..."):
            df = fetch_reviews(place_id)

        if df.empty:
            st.warning("No reviews fetched. Please check the Place ID or ensure your SerpApi key is valid.")
            return # Stop processing if no reviews

        st.success(f"Fetched {len(df)} reviews.")

        # --- Convert Date BEFORE Enrichment ---
        st.info("Attempting Date Conversion on initial DataFrame...")
        print("--- Debug: Before Initial Date Conversion ---")
        print("Columns:", df.columns)
        print("Index:", df.index)
        print("Data types:\n", df.dtypes)
        if 'Date' in df.columns:
            print("Sample 'Date' values:\n", df['Date'].head())
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                st.success("Initial Date column converted successfully.")
                print("--- Debug: After Initial Date Conversion ---")
                print("Data types:\n", df.dtypes)
                print("Sample 'Date' values:\n", df['Date'].head())
                print("-----------------------------------------")
            except Exception as e:
                st.error(f"Error during initial date conversion: {e}")
                print(f"Error during initial pd.to_datetime: {e}")
                import traceback
                traceback.print_exc()
                return # Stop if initial conversion fails
        else:
            st.error("Initial DataFrame is missing the 'Date' column.")
            print("'Date' column missing in initial df.")
            return # Stop if date column is missing initially

        # --- Enrich Reviews (using the original df and batch function) ---
        with st.spinner("Analyzing reviews with Gemini (in batches)..."):
            enriched_df = enrich_reviews_batch(df) # Pass the original df

        # --- Reset Index (Potentially less critical now, but keep for safety) ---
        # Explicitly reset the index to avoid potential index-related issues
        if not enriched_df.empty:
             try:
                 enriched_df.reset_index(drop=True, inplace=True)
                 print("--- Debug: After Resetting Index ---")
                 print("Index:", enriched_df.index)
                 print("------------------------------------")
             except Exception as e:
                 st.error(f"Error resetting index: {e}")
                 # Decide if you want to return here or continue cautiously

        if enriched_df.empty and not df.empty:
             st.warning("Review enrichment resulted in an empty dataset. This might indicate an issue with the Gemini API or response parsing.")
             # Optionally display the original df for debugging
             # st.dataframe(df)
             return # Stop if enrichment failed unexpectedly

        # --- Post-Enrichment Checks ---
        st.info("DataFrame state after batch enrichment:")
        st.dataframe(enriched_df) # Show DF after enrichment
        print("--- Debug: After Batch Enrichment ---")
        print("Columns:", enriched_df.columns)
        print("Index:", enriched_df.index)
        print("Data types:\n", enriched_df.dtypes)
        if 'Date' in enriched_df.columns:
            print("Sample 'Date' values after enrichment:\n", enriched_df['Date'].head())
        else:
            # This would be very bad if it happened now
            st.error("Critical Error: 'Date' column lost during enrichment process!")
            print("'Date' column MISSING after enrichment.")
            return
        print("-----------------------------")

        # --- Continue Processing (No date conversion needed here anymore) ---
        try:
            with st.spinner("Saving enriched reviews to Excel..."):
                enriched_df.to_excel("google_reviews_enriched.xlsx", index=False)
            st.success("Enriched reviews saved to google_reviews_enriched.xlsx")

            with st.spinner("Generating summary..."):
                summary = generate_summary(enriched_df)

            run_streamlit_app(enriched_df, summary)

        except Exception as e:
            st.error(f"An error occurred during processing after date conversion: {e}")
            st.dataframe(enriched_df) # Show dataframe state at point of error


if __name__ == "__main__":
    main()
