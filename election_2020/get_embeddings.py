import os
import pandas as pd
import json


from election_2020.openai_service import split_into_batches, dynamic_batches, OpenAIService
from election_2020 import DATA_DIRPATH, ELECT_FIRPATH
#save files in election folder as a sub folder under data


print("-----------------")
filename = 'user_sample_testing200.csv' ##for testing code
#filename = 'openai_user_sample_max_50_size_10000.csv'
filepath = os.path.join(ELECT_FIRPATH, filename)
#print(filepath)
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    print(df.shape) #make sure we have read the file
else:
    print(f"File not found: {filepath}")

print("-----------------")
print("EMBEDDINGS:")
ai = OpenAIService()
texts = df['tweet_texts'].tolist()
#get embeddings with dynamic batches
embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)

df = df.assign(openai_embeddings=embeddings)
print(df.shape) #check if we add the embedding column


print("-----------------")
print("SAVING...")
#save embeddings files back to the 'election 'folder and add a prefix 'embedding'
embeddings_csv_filepath = os.path.join(ELECT_FIRPATH, f"embedding_{filename}")
df.to_csv(embeddings_csv_filepath,index=False)