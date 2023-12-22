import os
import pandas as pd
import json

from election_2020.model_storage import ModelStorage
from election_2020 import DATA_DIRPATH, ELECT_FIRPATH

print("-------Load Embeddings---------")
#filename = 'embedding_user_sample_testing200.csv' #small sample file for test the code
filename = 'embedding_openai_user_sample_max_50_size_10000.csv'
filepath = os.path.join(ELECT_FIRPATH, filename)
#print(filepath)
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    print(df.shape) #make sure we have read the file
else:
    print(f"File not found: {filepath}")


#unpack
print("-------Unpack Embeddings---------")
def unpack(embeddings_str):
    # idempotence check
    if isinstance(embeddings_str, str):
        return json.loads(embeddings_str)
    else:
        return embeddings_str

#text_df used for saving prediction result, keep user_id for mapping users
text_df = df[['user_id','tweet_texts','openai_embeddings']]
text_df['openai_embeddings'] = text_df['openai_embeddings'].apply(unpack)

#embeds_df further used for X in the prediction
embeds_df = pd.DataFrame(text_df["openai_embeddings"].values.tolist())
embeds_df.columns = [str(i) for i in range(0, len(embeds_df.columns))]
embeds_df = text_df.drop(columns=["openai_embeddings"]).merge(embeds_df, left_index=True, right_index=True)
print(embeds_df.shape)

X = embeds_df.loc[:,'0':]

#load models
print("-------Load Model---------")
#y_col = "opinion_community" #@param ["is_bot", "opinion_community", "is_bom_overall", "is_bom_astroturf", "is_toxic", "is_factual",  "fourway_label"]
#model_dirname = "logistic_regression" #@param ["logistic_regression", "random_forest", "xgboost"]
#
#local_dirpath = f"/results/classification/{y_col}/{model_dirname}"
#model_storage = ModelStorage(local_dirpath)
#
#try:
#    model = model_storage.download_model()
#    print('done')
#except:
#    print('fail to download the model')

model_dirname = "logistic_regression"
for y_col in ["is_bot", "opinion_community", "is_toxic", "is_factual",  "fourway_label"]:
    local_dirpath = f"/results/classification/{y_col}/{model_dirname}"
    model = ModelStorage(local_dirpath).download_model()
    model_preds = model.predict(X)
    column_name = f"pred_{y_col}"
    text_df[column_name] = model_preds




#make prediction
#print("-------Prediction---------")
#
#model_preds = model.predict(X)
#print(model_preds[:10])
#column_name = f"{model_dirname}_{y_col}"
#text_df[column_name] = model_preds

saving_csv_filepath = os.path.join(ELECT_FIRPATH, f"LR_{filename}")
text_df.to_csv(saving_csv_filepath,index=False)

