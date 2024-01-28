import os
import pandas as pd
import json

from app.bq_service import BigQueryService
from app.model_storage import ModelStorage

import warnings
warnings.filterwarnings('ignore')

bq = BigQueryService()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

print("LOAD EMBEDDINGS:")
sql = '''
SELECT emb.user_id,
  emb.openai_embeddings
FROM `tweet-research-shared.election_2020_transition_2021_combined.openai_embeddings_max_50` emb 
LEFT JOIN `tweet-research-shared.election_2020_transition_2021_combined.LR_pred` p 
ON emb.user_id = p.user_id
WHERE p.user_id IS NULL
'''

df = bq.query_to_df(sql,verbose=False)
print(f"{len(df)} embeddings records")

#unpack
print("UNPACK")
#do not need this unpack function since we store embeddings as array of float
#def unpack(embeddings_str):
#    # idempotence check
#    if isinstance(embeddings_str, str):
#        return json.loads(embeddings_str)
#    else:
#        return embeddings_str

#df['openai_embeddings'] = df['openai_embeddings'].apply(unpack)

#embeds_df further used for X in the prediction
embeds_df = pd.DataFrame(df["openai_embeddings"].values.tolist())
embeds_df.columns = [str(i) for i in range(0, len(embeds_df.columns))]
embeds_df = df.drop(columns=["openai_embeddings"]).merge(embeds_df, left_index=True, right_index=True)
print(embeds_df.shape)

X = embeds_df.loc[:,'0':]

#load models
print("-------Load Model---------")

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

#reference for label dictionary: https://github.com/s2t2/openai-embeddings-2023/blob/main/results/tfidf_classification/fourway_label/logistic_regression/results.json

fourway_dict = {
    0: "Anti-Trump Bot",
    1: "Anti-Trump Human",
    2: "Pro-Trump Bot",
    3: "Pro-Trump Human"
}

df['fourway_label'] = df['fourway_label_pred'].map(fourway_dict)

df.drop(columns=['openai_embeddings'],inplace=True)
print(df.columns)

print('SAVING:')
#append predictions to predictions table
pred_table = f"{PROJECT_ID}.{DATASET_ID}.LR_pred"
df.to_gbq(pred_table,project_id=PROJECT_ID, if_exists='append')
