### Prep: BigQuery Tables

Create the table for the storage of user info:
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.user_tweets_sample_max_50`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.user_tweets_sample_max_50` AS (
    SELECT 
      user_id,
      COUNT(DISTINCT status_id) AS tweet_count,
      COUNT(DISTINCT CASE WHEN retweeted_status_id IS NOT NULL THEN status_id END) AS rt_count,
      MIN(DATE(created_at)) AS first_tweet_on,
      MAX(DATE(created_at)) AS latest_tweet_on,
      --STRING_AGG(status_text, {TWEET_DELIMETER} ORDER BY RAND() LIMIT {TWEET_MAX}) AS tweet_texts
      STRING_AGG(status_text, " || " ORDER BY RAND() LIMIT 50) AS tweet_texts
    FROM `tweet-research-shared.election_2020_transition_2021_combined.tweets_v2_slim`
    GROUP BY 1
)
```

Create table for the storage of embeddings:
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_embeddings_max_50`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.openai_embeddings_max_50` (

    user_id	INT64,
    openai_embeddings	ARRAY<FLOAT64>
)
```

Table for storage of predictions:
```sql
DROP TABLE IF EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_pred`;
CREATE TABLE IF NOT EXISTS `tweet-research-shared.election_2020_transition_2021_combined.LR_pred` (
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
    fourway_label_proba ARRAY<FLOAT64>

)
```

### Application

BigQuery Service: for the purpose of extracting data with SQL queries from BigQuery
```sh
python -m app.bq_service
```

Get Embeddings:
Run the following command if you want to get a sample of users and get embeddings for their tweets
```sh
python -m election_2020.get_embeddings
```

Classify Users:
Using the pre-trained Logistic Regression models, classify user
```sh
python -m election_2020.classify_user
```

