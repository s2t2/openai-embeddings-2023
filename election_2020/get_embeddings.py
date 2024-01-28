##file name lower case bq_service.py
## call it get embeddings
import os
import pandas as pd

from app.bq_service import BigQueryService
from app.openai_service import split_into_batches, dynamic_batches, OpenAIService

import warnings
warnings.filterwarnings('ignore')

bq = BigQueryService()
ai = OpenAIService()


print("SAMPLING USERS:")

#fetch new batch of sample users
TWEET_MAX = 50
TWEET_DELIMETER = "' || '"
USERS_LIMIT = os.getenv("USERS_LIMIT")
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

sample_sql = f'''
SELECT u.user_id, 
  u.tweet_count, 
  u.rt_count, 
  u.first_tweet_on, 
  u.latest_tweet_on, 
  u.tweet_texts
FROM `tweet-research-shared.election_2020_transition_2021_combined.user_tweets_sample_max_50` u
LEFT JOIN `tweet-research-shared.election_2020_transition_2021_combined.openai_embeddings_max_50` emb 
ON u.user_id = emb.user_id
WHERE emb.openai_embeddings IS NULL 
ORDER BY RAND()
LIMIT {USERS_LIMIT}
'''
#can add a limit on min tweet_count to filter users later

sample = bq.query_to_df(sample_sql, verbose=False)

#get embeddings with openai service
print("ENBEDDINGS:")
texts = sample['tweet_texts'].tolist()
embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)

#the embeddings table has two field: user_id AS INT, openai_embedding AS ARRAY
sample_embeddings = sample[['user_id']]
sample_embeddings['openai_embeddings'] = embeddings

#save the embeddings back to bigquery
embeddings_table = f"{PROJECT_ID}.{DATASET_ID}.openai_embeddings_max_50"
sample_embeddings.to_gbq(embeddings_table, project_id=PROJECT_ID, if_exists='append')

