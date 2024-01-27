

import os
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.openai_service import OpenAIService


load_dotenv()

TEXTS_LIMIT = os.getenv("TEXTS_LIMIT")


if __name__ == "__main__":

    bq = BigQueryService()
    print(bq)
    print("DATASET ADDRESS:", bq.dataset_address)

    print("---------------")
    print("TEXTS...")
    #print("LIMIT: ", TEXTS_LIMIT)

    sql = f"""
        -- FETCH STATUSES WE HAVEN'T ALREADY RETRIEVED EMBEDDINGS FOR
        SELECT DISTINCT txt.status_text_id, txt.status_text --, emb.status_text_id
        FROM `{bq.dataset_address}.botometer_sample_max_50_texts_map` txt
        LEFT JOIN  `{bq.dataset_address}.botometer_sample_max_50_openai_text_embeddings` emb
            ON emb.status_text_id = txt.status_text_id
        WHERE emb.status_text_id IS NULL
        ORDER BY txt.status_text_id
    """

    if TEXTS_LIMIT:
        texts_limit = int(TEXTS_LIMIT)
        sql += f"    LIMIT {texts_limit} "

    df = bq.query_to_df(sql)

    #if TEXTS_LIMIT:
    #    texts_limit = int(TEXTS_LIMIT)
    #    df = df.iloc[0:texts_limit]

    print("---------------")
    print("EMBEDDINGS...")
    texts = df["status_text"].tolist()

    ai = OpenAIService()
    embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)
    #print(len(embeddings))

    df["embeddings"] = embeddings
    records = df[["status_text_id", "embeddings"]].to_dict("records")

    print("---------------")
    print("SAVING...")

    embeddings_table = bq.client.get_table(f"{bq.dataset_address}.botometer_sample_max_50_openai_text_embeddings") # API call!
    errors = bq.insert_records_in_batches(embeddings_table, records, batch_size=50) # running into google api issues with larger batches - there are so many embeddings for each row, so we lower the batch count substantially
    if any(errors):
        print("ERRORS:")
        print(errors)

    print("---------------")
    print("JOB COMPLETE!")
