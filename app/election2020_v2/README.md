
Use election_transaction_2020 dataset, get embeddings on both user level and tweet level, and apply the pre-trained classification models on the dataset.

## Google BigQuery Setup
Create several tables in Google BigQuery for further execution

Create table for user and tweets sample:
```sql
CREATE TABLE `tweet-research-shared.election_2020_transition_2021_combined.openai_user_tweets_sample_v2` AS (
  SELECT user_id,
    ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY RAND()) AS row_num,
    status_id,
    status_text,
    created_at
  FROM `tweet-research-shared.election_2020_transition_2021_combined.tweets_v2_slim`
  ORDER BY 1,5
);
```

Sample with max limit 50:
```sql
CREATE TABLE `tweet-research-shared.election_2020_transition_2021_combined.openai_user_tweets_sample_max50_v2` AS (
  SELECT *
  FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_user_tweets_sample_v2`
  WHERE row_num <= 50
  ORDER BY user_id, row_num
);
```

Users and tweets under max50 limit
```sql
SELECT COUNT(DISTINCT user_id), --3364945
  COUNT(DISTINCT status_id) --15622828
FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_user_tweets_sample_max50_v2`
```

Status text sample for embeddings per tweet
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_text_sample_max50`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_text_sample_max50` AS(
  SELECT
    ROW_NUMBER() OVER() AS status_text_id,
    status_text,
    COUNT(DISTINCT status_id) AS status_count,
    ARRAY_AGG(DISTINCT status_id) AS status_ids,
    COUNT(DISTINCT user_id) AS user_count,
    ARRAY_AGG(DISTINCT user_id) AS user_ids
  FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_user_tweets_sample_v2`
  WHERE row_num <= 50
  GROUP BY 2
  ORDER BY 1
);
```

table for embeddings storage:
```sql
--table for user embeddings
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_user_embeddings`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_user_embeddings` (
  user_id INT64,
  embeddings ARRAY<FLOAT64>
);

--table for tweet embeddings
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_tweet_embeddings`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_tweet_embeddings` (
  status_text_id INT64,
  embeddings ARRAY<FLOAT64>
);
```

Tables for classification results:
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_user_pred`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_user_pred` (
    user_id	INT64,

    is_bot_pred BOOL,
    is_bot_proba ARRAY<FLOAT64>,

    opinion_community_pred INT64,
    opinion_community_proba ARRAY<FLOAT64>,

    is_toxic_pred INT64,
    is_toxic_proba ARRAY<FLOAT64>,

    is_factual_pred FLOAT64,
    is_factual_proba ARRAY<FLOAT64>,

    fourway_label_pred INT64,
    fourway_label_proba ARRAY<FLOAT64>,
    fourway_label STRING
);


DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_tweet_pred`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_tweet_pred` (
    status_text_id	INT64,

    is_bot_pred BOOL,
    is_bot_proba ARRAY<FLOAT64>,

    opinion_community_pred INT64,
    opinion_community_proba ARRAY<FLOAT64>,

    is_toxic_pred INT64,
    is_toxic_proba ARRAY<FLOAT64>,

    is_factual_pred FLOAT64,
    is_factual_proba ARRAY<FLOAT64>,

    fourway_label_pred INT64,
    fourway_label_proba ARRAY<FLOAT64>,
    fourway_label STRING
);
```

## Embeddings

Embeddings per user:
```sh
python -m app.election2020_v2.openai_per_user

USERS_LIMIT=10 python -m app.election2020_v2.openai_per_user
```

Embeddings per tweet:
```sh
python -m app.election2020_v2.openai_per_tweet

TWEETS_LIMIT=10 python -m app.election2020_v2.openai_per_tweet
```

Mapping embeddings per tweet with status_id:
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.text_tweet_mapping`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.text_tweet_mapping` (
  WITH lookup_tb AS (
    SELECT txt.status_text_id, status_id
    FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_text_sample_max50` txt,
    UNNEST(txt.status_ids) as status_id
  )

  SELECT tb.status_id, tb.status_text_id, emb.embeddings
  FROM lookup_tb tb
  JOIN `tweet-research-shared.election_2020_transition_2021_combined.openai_tweet_embeddings` emb
  ON tb.status_text_id = emb.status_text_id
  ORDER BY 2
);
```


## Prediction:

Apply pre-trained Logistic Regression models on the user/tweet embeddings
```sh
python -m app.election2020_v2.predict_user
```

```sh
python -m app.election2020_v2.predict_tweet
```