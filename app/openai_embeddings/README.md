# OpenAI Embeddings

Get embeddings, not only per user, but also per tweet, so we can compare the two approaches. Pull a new sample of tweets for the users we have been analyzing, but this time make sure to keep track of which tweets are being used, which will aid comparisons.

## Setup

Migrations, as necessary. Here we create a table of all tweets from each user in the sample:

```sql
--CREATE TABLE `tweet-collector-py.impeachment_production.botometer_sample_max_10` as (
--CREATE TABLE `tweet-collector-py.impeachment_production.botometer_sample_max_50` as (
CREATE TABLE `tweet-collector-py.impeachment_production.botometer_sample` as (
    WITH ranked_tweets AS (
      SELECT
          u.user_id, t.status_id, t.status_text, t.created_at,
          ROW_NUMBER() OVER (PARTITION BY u.user_id ORDER BY RAND()) AS row_num
      FROM (
        SELECT DISTINCT user_id
        FROM `tweet-collector-py.impeachment_production.botometer_sample_openai_tweet_embeddings_20230724`
      ) u
      JOIN `tweet-collector-py.impeachment_production.tweets_v2` t on t.user_id = u.user_id
      ORDER BY u.user_id, t.created_at
      --LIMIT 10
    )

    SELECT user_id, row_num,
        status_id, status_text, created_at,
    FROM ranked_tweets
    -- WHERE row_num <= 10 -- MAX_TWEETS_PER_USER
    -- WHERE row_num <= 50 -- MAX_TWEETS_PER_USER

);
```


How to sample from this table (choose a `MAX_TWEETS_PER_USER`, which we set as 50 by default):

```sql
SELECT
  count(distinct user_id) as user_count -- 7566
  ,count(distinct status_id) as status_count -- 183727
FROM `tweet-collector-py.impeachment_production.botometer_sample`
WHERE row_num <= 50 -- MAX_TWEETS_PER_USER
```

The 7,566 users in this sample have 183,727 tweets.

Unique table of texts with identifiers:

```sql
DROP TABLE IF EXISTS `tweet-collector-py.impeachment_production.botometer_sample_max_50_texts_map`;
CREATE TABLE IF NOT EXISTS `tweet-collector-py.impeachment_production.botometer_sample_max_50_texts_map` as (
    --WITH texts_map as (
        SELECT --s.user_id, s.row_num, s.status_id, s.status_text, s.created_at
            ROW_NUMBER() OVER () AS status_text_id
            ,s.status_text
            ,count(DISTINCT s.status_id) as status_count
            ,array_agg(DISTINCT s.status_id) as status_ids
            ,count(DISTINCT s.user_id) as user_count
            ,array_agg(DISTINCT s.user_id) as user_ids
        FROM `tweet-collector-py.impeachment_production.botometer_sample` s
        WHERE s.row_num <= 50 -- MAX_TWEETS_PER_USER
        GROUP BY 2
        --ORDER BY status_count desc
    --)
    --SELECT status_text, status_count, status_id
    --FROM texts_map,
    --UNNEST(status_ids) AS status_id
)
```

Of the 183,727 tweets in this sample, there are 80,205 unique texts.

Migrate table to receive text embeddings:

```sql
CREATE TABLE IF NOT EXISTS `tweet-collector-py.impeachment_production.botometer_sample_max_50_openai_text_embeddings` (
    status_text_id	INT64,
    embeddings ARRAY<FLOAT64>
)
```

Migrate table to receive user embeddings:

```sql
DROP TABLE IF EXISTS `tweet-collector-py.impeachment_production.botometer_sample_max_50_openai_user_embeddings`;
CREATE TABLE IF NOT EXISTS `tweet-collector-py.impeachment_production.botometer_sample_max_50_openai_user_embeddings` (
    user_id	    INT64,
    embeddings ARRAY<FLOAT64>
)
```


## Fetch Embeddings (Per Tweet)

Fetch tweet-level embeddings, and store in BQ:

```sh
python -m app.openai_embeddings.per_tweet.embeddings_job

TEXTS_LIMIT=10 python -m app.openai_embeddings.per_tweet
TEXTS_LIMIT=1500 python -m app.openai_embeddings.per_tweet
TEXTS_LIMIT=10000 python -m app.openai_embeddings.per_tweet
TEXTS_LIMIT=250000 python -m app.openai_embeddings.per_tweet
```

Monitoring the results:

```sql
SELECT count(distinct status_text_id) as text_count
FROM `tweet-collector-py.impeachment_production.botometer_sample_max_50_openai_text_embeddings`  emb
```

## Fetch Embeddings (Per User)

Fetch user-level embeddings, and store in BQ:

```sh
python -m app.openai_embeddings.per_user

USERS_LIMIT=10 python -m app.openai_embeddings.per_user
USERS_LIMIT=100 python -m app.openai_embeddings.per_user
USERS_LIMIT=1000 python -m app.openai_embeddings.per_user
```

Monitoring the results:

```sql
SELECT
    count(distinct s.user_id) as user_count
    ,count(distinct case when emb.user_id is not null then s.user_id end) as users_collected
    ,count(distinct case when emb.user_id is not null then s.user_id end) / count(distinct s.user_id) as pct_collected
FROM `tweet-collector-py.impeachment_production.botometer_sample` s
LEFT JOIN `tweet-collector-py.impeachment_production.botometer_sample_max_50_openai_user_embeddings`  emb
  ON s.user_id = emb.user_id

```
