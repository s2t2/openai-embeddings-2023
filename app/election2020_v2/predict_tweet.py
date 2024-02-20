import os
import pandas as pd
from dotenv import load_dotenv

from app.bq_service import BigQueryService
from app.model_storage import ModelStorage

import warnings
warnings.filterwarnings('ignore')

load_dotenv()
TEXTS_LIMIT = os.getenv('TEXTS_LIMIT')

if __name__ == "__main__":
    bq = BigQueryService()
    print(bq)
    print(f"DATASET: {bq.dataset_address}")

    print("---------------")
    print("LOADING EMBEDDINGS:")

    sql =  f'''
    SELECT emb.status_text_id,
        emb.embeddings
    FROM `{bq.dataset_address}.openai_tweet_embeddings` emb 
    LEFT JOIN `{bq.dataset_address}.openai_tweet_embeddings_lr_preds` tp
    ON emb.status_text_id = tp.status_text_id
    WHERE tp.status_text_id IS NULL
    '''

    if TEXTS_LIMIT:
        text_limit = int(TEXTS_LIMIT)
        sql += f"  LIMIT {text_limit}"

    df = bq.query_to_df(sql, verbose=False)
    print(len(df))
    if df.empty:
        print("NO MORE TWEETS TO MAKE PREDICTION... GOODBYE!")
        exit()

    embeds_df = pd.DataFrame(df["embeddings"].values.tolist())
    embeds_df.columns = [str(i) for i in range(0, len(embeds_df.columns))]
    embeds_df = df.drop(columns=["embeddings"]).merge(embeds_df, left_index=True, right_index=True)
    #print(embeds_df.shape)

    X = embeds_df.loc[:,'0':]

    print("---------------")
    print("MODELS PREDICTING:")

    model_dirname = "logistic_regression"
    for target in ["is_bot", "opinion_community", "is_toxic", "is_factual",  "fourway_label"]:
        local_dirpath = f"/results/classification/{target}/{model_dirname}"
        model = ModelStorage(local_dirpath).download_model()
        #predict
        model_preds = model.predict(X)
        pred_column_name = f"{target}_pred"
        df[pred_column_name] = model_preds
        #predict probabilities
        model_probas = model.predict_proba(X)
        proba_column_name = f"{target}_proba"
        df[proba_column_name] = model_probas.tolist()


    fourway_dict = {
        0: "Anti-Trump Bot",
        1: "Anti-Trump Human",
        2: "Pro-Trump Bot",
        3: "Pro-Trump Human"
    }

    df['fourway_label'] = df['fourway_label_pred'].map(fourway_dict)

    df.drop(columns=['embeddings'],inplace=True)

    records = df.to_dict("records")

    print("---------------")
    print("SAVING:")

    pred_table_name = f"{bq.dataset_address}.openai_tweet_embeddings_lr_preds"
    pred_table = bq.client.get_table(pred_table_name) 
    errors = bq.insert_records_in_batches(pred_table, records, batch_size=50) 
    if any(errors):
        print("ERRORS:")
        print(errors)

    print("---------------")
    print("JOB COMPLETE!")
    
