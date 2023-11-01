from google.cloud import bigquery

import os
from dotenv import load_dotenv

# Load .env file 
load_dotenv()

# Get GCP's secrets
KEYS_FILE = os.getenv("KEYS_FILE")

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEYS_FILE

# Initialize the BigQuery client
client = bigquery.Client()

# Write your BigQuery SQL query
query = """
WITH LabCounts AS (
    SELECT hadm_id, itemid, COUNT(*) AS lab_count
    FROM `physionet-data.mimiciv_hosp.labevents`
    GROUP BY hadm_id, itemid
    HAVING COUNT(*) >= 500
)

SELECT l.hadm_id, l.itemid, l.charttime, l.value, l.valuenum, l.valueuom, l.ref_range_lower, l.ref_range_upper, l.flag, l.priority, l.comments
FROM `physionet-data.mimiciv_hosp.labevents` AS l
INNER JOIN LabCounts AS c
ON l.hadm_id = c.hadm_id AND l.itemid = c.itemid
ORDER BY l.hadm_id, l.charttime;
"""

# Run the query
query_job = client.query(query)

# Wait for the query to complete
results = query_job.result()

# Convert the results to a Pandas DataFrame
df = results.to_dataframe()