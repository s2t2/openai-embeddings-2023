import os
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.openai_service import split_into_batches, dynamic_batches, OpenAIService

import warnings
warnings.filterwarnings('ignore')

load_dotenv()
USERS_LIMIT = os.getenv('USERS_LIMIT')
MAX_TWEETS_PER_USER = 50

if __name__ == "__main__":
    bq = BigQueryService()
    print(bq)
    print(f"DATASET: {bq.dataset_address}")

    print("---------------")
    print("FETCHING USER:")

    sql = f'''
        WITH user_sample AS(
        SELECT user_id, 
          MIN(row_num) AS row_min,
          MAX(row_num) AS row_max,
          COUNT(DISTINCT status_id) AS status_count_max_50,
          ARRAY_AGG(status_id) AS status_ids,
          STRING_AGG(status_text) AS status_texts
        FROM `{bq.dataset_address}.openai_user_tweets_sample_max50_v2`
        GROUP BY 1
        )

        SELECT u.user_id, u.status_count_max_50, u.status_ids, u.status_texts
        FROM user_sample u
        LEFT JOIN  `{bq.dataset_address}.openai_user_embeddings` emb
        ON u.user_id = emb.user_id
        WHERE emb.user_id IS NULL
        ORDER BY u.user_id
    '''

    if USERS_LIMIT:
        users_limit = int(USERS_LIMIT)
        sql += f"  LIMIT {users_limit}"

    df = bq.query_to_df(sql, verbose=False)
    print(len(df))
    if df.empty:
        print("NO MORE USERS TO PROCESS... GOODBYE!")
        exit()

    print("---------------")
    print("EMBEDDINGS:")
    ai = OpenAIService()

    texts = df['status_texts'].tolist()
    embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)

    df['embeddings'] = embeddings
    records = df[['user_id','embeddings']].to_dict("records")

    print("---------------")
    print("SAVING:")

    embeddings_table_name = f"{bq.dataset_address}.openai_user_embeddings"
    embeddings_table = bq.client.get_table(embeddings_table_name) 
    errors = bq.insert_records_in_batches(embeddings_table, records, batch_size=50) 
    if any(errors):
        print("ERRORS:")
        print(errors)

    print("---------------")
    print("JOB COMPLETE!")


