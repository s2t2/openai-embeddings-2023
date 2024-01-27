


from app.bq_service import BigQueryService



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

    df = bq.query_to_df(sql)

    breakpoint()
