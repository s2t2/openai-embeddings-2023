import pandas as pd

from app.BigQuery_service import BigQueryService
from app.openai_service import split_into_batches, dynamic_batches, OpenAIService
from app import DATA_DIRPATH, ELECT_FIRPATH

bq = BigQueryService()
ai = OpenAIService()


#get the user sample from bigquery with sql
#query the existing user from bq dataset
print("SAMPLING USERS:")
sql = '''
SELECT user_id
FROM `tweet-research-shared.election_2020_transition_2021_combined.users_sample_max50` 
'''
exist_users = bq.query_to_df(sql, verbose=False)
exist_user_id = tuple(exist_users['user_id'].tolist())
print("{} users exist".format(len(exist_users)))

#fetch new batch of sample users
TWEET_MAX = 50
TWEET_DELIMETER = "' || '"
SAMPLE_SIZE = input("Please input a sample size:") or 100

sample_sql = f'''
SELECT user_id,
  COUNT(DISTINCT status_id) AS tweet_counts,
  COUNT(DISTINCT CASE WHEN retweeted_status_id IS NOT NULL THEN status_id END) AS rt_counts,
  MIN(DATE(created_at)) AS first_tweet_on,
  MAX(DATE(created_at)) AS latest_tweet_on,
  STRING_AGG(status_text, {TWEET_DELIMETER} ORDER BY RAND() LIMIT {TWEET_MAX}) AS tweet_texts
  FROM `tweet-research-shared.election_2020_transition_2021_combined.tweets_v2_slim`
  WHERE user_id NOT IN {exist_user_id}
  GROUP BY 1
  ORDER BY RAND()
  LIMIT {SAMPLE_SIZE}
'''

sample = bq.query_to_df(sample_sql, verbose=False)

#convert dtype
sample['first_tweet_on'] = pd.to_datetime(sample['first_tweet_on'])
sample['latest_tweet_on'] = pd.to_datetime(sample['latest_tweet_on'])

#append users to user sample table
project_id = "tweet-research-shared"
dataset_id = "election_2020_transition_2021_combined"
user_table = f"{project_id}.{dataset_id}.users_sample_max50"
sample.to_gbq(user_table,project_id=project_id, if_exists='append')


#get embeddings with openai service
print("ENBEDDINGS:")
texts = sample['tweet_texts'].tolist()
embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)
sample_embeddings = sample.assign(openai_embeddings=embeddings)

sample_embeddings['openai_embeddings'] = sample_embeddings['openai_embeddings'].astype('str')

#save the embeddings back to bigquery
embeddings_table = f"{project_id}.{dataset_id}.openai_embeddings_users_sample_max50"
sample_embeddings.to_gbq(embeddings_table, project_id=project_id, if_exists='append')

