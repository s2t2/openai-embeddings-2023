

import os
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.openai_service import OpenAIService


load_dotenv()

USERS_LIMIT = os.getenv("USERS_LIMIT")

MAX_TWEETS_PER_USER = 50


if __name__ == "__main__":

    bq = BigQueryService()
    print(bq)
    print("DATASET ADDRESS:", bq.dataset_address)

    print("---------------")
    print("USERS...")
    #print("LIMIT: ", USERS_LIMIT)

    sql = f"""
        -- FETCH USERS WE HAVEN'T ALREADY RETRIEVED EMBEDDINGS FOR

        WITH users_sample as (
            SELECT
                s.user_id --,min(s.row_num) as row_min, max(s.row_num) as row_max
                ,count(distinct s.status_id) as status_count_max_50
                ,array_agg(distinct s.status_id) as status_ids
                ,string_agg(distinct s.status_text, " ") as status_texts
            FROM `{bq.dataset_address}.botometer_sample` s
            WHERE s.row_num <= {int(MAX_TWEETS_PER_USER)}
            GROUP BY 1
            -- ORDER BY user_id
        )

        SELECT u.user_id, u.status_count_max_50, u.status_ids, u.status_texts
        FROM users_sample u
        LEFT JOIN  `{bq.dataset_address}.botometer_sample_max_50_openai_user_embeddings` emb
            ON u.user_id = emb.user_id
        WHERE emb.user_id IS NULL
        ORDER BY u.user_id
    """

    if USERS_LIMIT:
        users_limit = int(USERS_LIMIT)
        sql += f"    LIMIT {users_limit} "

    df = bq.query_to_df(sql)
    print(len(df))
    if df.empty:
        print("NO MORE USERS TO PROCESS... GOODBYE!")
        exit()

    print("---------------")
    print("EMBEDDINGS...")
    texts = df["status_texts"].tolist()

    ai = OpenAIService()
    embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)
    #print(len(embeddings))

    df["embeddings"] = embeddings
    records = df[["user_id", "embeddings"]].to_dict("records")

    print("---------------")
    print("SAVING...")

    embeddings_table_name = f"{bq.dataset_address}.botometer_sample_max_50_openai_user_embeddings"
    embeddings_table = bq.client.get_table(embeddings_table_name) # API call!
    errors = bq.insert_records_in_batches(embeddings_table, records, batch_size=50) # running into google api issues with larger batches - there are so many embeddings for each row, so we lower the batch count substantially
    if any(errors):
        print("ERRORS:")
        print(errors)

    print("---------------")
    print("JOB COMPLETE!")
