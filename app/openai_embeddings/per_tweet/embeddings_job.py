

import os
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.openai_service import OpenAIService


load_dotenv()

TEXTS_LIMIT = os.getenv("TEXTS_LIMIT")


if __name__ == "__main__":

    bq = BigQueryService()
    print(bq)

    #sql = f"""
    #    SELECT DISTINCT status_text_id, status_text
    #    FROM `{bq.dataset_address}.botometer_sample_max_50_texts_map`
    #    ORDER BY status_text_id
    #"""

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
        sql += f" LIMIT {texts_limit} "

    df = bq.query_to_df(sql)

    #if TEXTS_LIMIT:
    #    texts_limit = int(TEXTS_LIMIT)
    #    df = df.iloc[0:texts_limit]

    print("EMBEDDINGS:")
    texts = df["status_text"].tolist()

    ai = OpenAIService()
    embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)
    #sample_embeddings = sample.assign(openai_embeddings=embeddings)
    #print(len(embeddings))

    #records = []
    #text_ids = df["status_text_id"].tolist()
    #for text_id, emb in zip(text_ids, embeddings):
    #    records.append({
    #        "status_text_id": text_id,
    #    })

    df["embeddings"] = embeddings
    records = df[["status_text_id", "embeddings"]].to_dict("records")

    embeddings_table = bq.client.get_table(f"{bq.dataset_address}.botometer_sample_max_50_openai_text_embeddings") # API call!
    errors = bq.insert_records_in_batches(embeddings_table, records)
    if any(errors):
        print("ERRORS:")
        print(errors)
