from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

credentials = service_account.Credentials.from_service_account_file(
'/Users/Joyce/Desktop/Project/openai-embeddings-2023/google-credentials.json')



class BigQueryService():
    def __init__(self):
        self.client = bigquery.Client.from_service_account_json('/Users/Joyce/Desktop/Project/openai-embeddings-2023/google-credentials.json')

        #self.client = bigquery.Client(project="tweet-research-shared")

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

    sql = '''
    SELECT user_id
    FROM `tweet-research-shared.election_2020_transition_2021_combined.users_sample_max50_10000` 
    LIMIT 10
    '''

    users = bq.query_to_df(sql, verbose=False)
    print(users['user_id'].tolist())