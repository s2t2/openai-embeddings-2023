import os
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.openai_service import split_into_batches, dynamic_batches, OpenAIService

import warnings
warnings.filterwarnings('ignore')


load_dotenv()
TEXTS_LIMIT = os.getenv('TEXTS_LIMIT')

if __name__ == "__main__":
    bq = BigQueryService()
    print(bq)
    print(f"DATASET: {bq.dataset_address}")

    print("---------------")
    print("FETCHING TWEETS:")

    sql = f'''
    SELECT t.status_text_id,
      t.status_text
    FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_text_sample_max50` t
    LEFT JOIN `tweet-research-shared.election_2020_transition_2021_combined.openai_tweet_embeddings` emb 
    ON t.status_text_id = emb.status_text_id
    WHERE emb.status_text_id IS NULL
    ORDER BY 1
    '''

    if TEXTS_LIMIT:
        text_limit = int(TEXTS_LIMIT)
        sql += f"  LIMIT {text_limit}"
    
    df = bq.query_to_df(sql, verbose=False)
    print(len(df))
    if df.empty:
        print("NO MORE TEXTS TO PROCESS... GOODBYE!")
        exit()


    print("---------------")
    print("EMBEDDINGS:")
    ai = OpenAIService()

    texts = df['status_text'].tolist()
    embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)

    df['embeddings'] = embeddings
    records = df[['status_text_id','embeddings']].to_dict("records")

    print("---------------")
    print("SAVING:")

    embeddings_table_name = f"tweet-research-shared.election_2020_transition_2021_combined.openai_tweet_embeddings"
    embeddings_table = bq.client.get_table(embeddings_table_name) 
    errors = bq.insert_records_in_batches(embeddings_table, records, batch_size=50) 
    if any(errors):
        print("ERRORS:")
        print(errors)

    print("---------------")
    print("JOB COMPLETE!")