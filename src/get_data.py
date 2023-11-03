from google.cloud import bigquery

import os
from dotenv import load_dotenv

# Load .env file 
load_dotenv()

# Get GCP keys file path
KEYS_FILE = os.getenv("KEYS_FILE")

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEYS_FILE

# Initialize the BigQuery client
client = bigquery.Client()

# Write the SQL query

## OPTION 1: Filter by grouping hadmid and itemid (Lab counts per admission 1 independently on how many times the lab was taken in the admission) 
# E.g. Lab taken 3 times in the same admission, counts as 1
# Filtering by 500: Returns 6188
# Filtering by 5: Returns 40,557,000 rows
query = """
WITH LabCounts AS (
    SELECT 
      hadm_id, 
      itemid, 
      COUNT(*) AS lab_count
    FROM 
      `physionet-data.mimiciv_hosp.labevents`
    GROUP BY 
    hadm_id, itemid
    HAVING 
      COUNT(*) >= 500
)

SELECT 
  l.hadm_id, 
  l.subject_id,
  l.itemid, 
  l.charttime, 
  l.storetime, 
  l.valuenum
FROM 
  `physionet-data.mimiciv_hosp.labevents` AS l
INNER JOIN 
  LabCounts AS c
ON 
  l.hadm_id = c.hadm_id AND l.itemid = c.itemid
ORDER BY 
  l.hadm_id, l.charttime;
"""

## OPTION 2: Filter by just itemid (Lab counts 1 every time lab was taken): 
# E.g. Lab taken 10 times in the same admission, counts as 10
# Filtering by 500: Returns 118,128,502 rows

query = """
WITH LabCodeCounts AS (
  SELECT
    itemid,
    COUNT(*) AS code_count
  FROM
    `physionet-data.mimiciv_hosp.labevents`
  GROUP BY
    itemid
  HAVING
    code_count > 500
)

SELECT
  le.hadm_id,
  le.subject_id,
  le.itemid,
  le.charttime,
  le.storetime,
  le.valuenum
FROM
  `physionet-data.mimiciv_hosp.labevents` AS le
JOIN
  LabCodeCounts AS lcc
ON
  le.itemid = lcc.itemid;
"""

# Run the query
query_job = client.query(query)

# Wait for the query to complete
results = query_job.result()

# Convert the results to a Pandas DataFrame
df = results.to_dataframe()