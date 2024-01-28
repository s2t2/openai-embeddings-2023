import pandas as pd
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CREDENTIALS_FILEPATH = os.getenv("GOOGLE_CREDENTIALS_FILEPATH")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

class BigQueryService():
    def __init__(self):
        self.client = bigquery.Client.from_service_account_json(GOOGLE_CREDENTIALS_FILEPATH)

    def execute_query(self, sql, verbose=True):
        if verbose == True:
            print(sql)
        job = self.client.query(sql)
        return job.result()

    def query_to_df(self, sql, verbose=True):
        """high-level wrapper to return a DataFrame"""
        results = self.execute_query(sql, verbose=verbose)
        records = [dict(row) for row in list(results)]
        df = pd.DataFrame(records)
        return df
    

if __name__ == "__main__":
    bq = BigQueryService()
    print(bq)

    #simple test to pull topics from topics dataset
    sql = f'''
    SELECT topic
    FROM `{PROJECT_ID}.{DATASET_ID}.topics` 
    '''

    topics = bq.query_to_df(sql, verbose=False)
    print(topics['topic'].tolist())