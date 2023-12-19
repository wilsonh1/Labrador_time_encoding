from google.cloud import bigquery

import os
from dotenv import load_dotenv

def get_df(query, save_csv=False, csv_name="data.csv"):
    """
    Get a Pandas DataFrame from a BigQuery query.

    params:
    query: str
      The query to run on BigQuery.
    save_csv: bool
      Whether to save the DataFrame to a CSV file.
    csv_name: str
      The name of the CSV file to save the DataFrame to.
    returns:
    df: Pandas DataFrame
      The DataFrame containing the results of the query.
    """
    # Load .env file 
    load_dotenv()

    # Get GCP keys file path
    KEYS_FILE = os.getenv("KEYS_FILE")

    # Set environment variables
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEYS_FILE

    # Initialize the BigQuery client
    client = bigquery.Client()

    # Run the query
    query_job = client.query(query)

    # Wait for the query to complete
    results = query_job.result()

    # Convert the results to a Pandas DataFrame
    df = results.to_dataframe()

    # Save the DataFrame to a CSV file
    if save_csv:
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", csv_name)
        df.to_csv(path, index=False)

    return df
