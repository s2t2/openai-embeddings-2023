# https://github.com/s2t2/openai-embeddings-2023/blob/main/notebooks/1_botometer_users_sample_and_openai_embeddings_20230704.py

import os
import pandas as pd
from time import sleep
from pprint import pprint
import json

import openai
from openai import Model, Embedding
from pandas import DataFrame
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("OPENAI_EMBEDDING_MODEL_ID", default="text-embedding-ada-002")

openai.api_key = OPENAI_API_KEY



def split_into_batches(my_list, batch_size=10_000):
    """Splits a list into evenly sized batches"""
    # h/t: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(my_list), batch_size):
        yield my_list[i : i + batch_size]

def dynamic_batches(texts, batch_char_limit=30_000):
    """Splits texts into batches, with specified max number of characters per batch.
        Caps text length at the maximum batch size (individual text cannot exceed batch size).
        Batches may have different lengths.
    """
    batches = []

    batch = []
    batch_chars = 0
    for text in texts:
        text_chars = len(text)

        if (batch_chars + text_chars) <= batch_char_limit:
            # THERE IS ROOM TO ADD THIS TEXT TO THE BATCH
            batch.append(text)
            batch_chars += text_chars
        else:
            # NO ROOM IN THIS BATCH, START A NEW ONE:

            if text_chars > batch_char_limit:
                # CAP THE TEXT AT THE MAX BATCH LENGTH
                text = text[0:batch_char_limit-1]

            batches.append(batch)
            batch = [text]
            batch_chars = text_chars

    if batch:
        batches.append(batch)

    return batches



class OpenAIService():
    """OpenAI API Service

        + https://github.com/openai/openai-python
        + https://platform.openai.com/account/api-keys
        + https://platform.openai.com/docs/introduction/key-concepts
        + https://platform.openai.com/docs/models/overview
        + https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
        + https://platform.openai.com/docs/guides/embeddings/embedding-models

        > We recommend using `text-embedding-ada-002` for nearly all
        (Embedding) use cases. It's better, cheaper, and simpler to use.
    """

    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        print("EMBEDDING MODEL:", self.model_id)


    def get_models(self):
        models = Model.list()
        #print(type(models)) #> openai.openai_object.OpenAIObject

        records = []
        for model in sorted(models.data, key=lambda m: m.id):
            #print(model.id, "...", model.owned_by, "...", model.parent, "...", model.object)
            model_info = model.to_dict()
            del model_info["permission"] # nested list
            #print(model_info)
            records.append(model_info)

        models_df = DataFrame(records)
        #models_df.to_csv("openai_models.csv")
        #models_df.sort_values(by=["id"])
        return models_df

    def get_embeddings(self, texts):
        """Pass in a list of strings. Returns a list of embeddings for each."""
        result = Embedding.create(input=texts, model=self.model_id) # API CALL
        #print(len(result["data"]))
        return [d["embedding"] for d in result["data"]]

    def get_embeddings_in_batches(self, texts, batch_size=250, sleep_seconds=60):
        """High level wrapper to work around RateLimitError:
                Rate limit reached for [MODEL] in [ORG] on tokens per min.
                Limit: 1_000_000 tokens / min.

            batch_size : Number of users to request per API call

            sleep : Wait for a minute before requesting the next batch

            Also beware InvalidRequestError:
                This model's maximum context length is 8191 tokens,
                however you requested X tokens (X in your prompt; 0 for the completion).
                Please reduce your prompt; or completion length.

            ... so we should make lots of smaller requests.
        """
        #embeddings = []
        #counter = 1
        #for texts_batch in split_into_batches(texts, batch_size=batch_size):
        #    print(counter, len(texts_batch))
        #    embeds_batch = self.get_embeddings(texts_batch) # API CALL
        #    embeddings += embeds_batch
        #    counter += 1
        #    sleep(sleep_seconds)
        #return embeddings

        #embeddings = []
        #counter = 1
        #for texts_batch in split_into_batches(texts, batch_size=batch_size):
        #    print(counter, len(texts_batch))
        #    try:
        #        embeds_batch = self.get_embeddings(texts_batch)  # API CALL
        #        embeddings += embeds_batch
        #    except openai.error.RateLimitError as err:
        #        print(f"Rate limit reached. Sleeping for {sleep_seconds} seconds.")
        #        sleep(sleep_seconds)
        #        continue
        #    counter += 1
        #return embeddings

        embeddings = []
        counter = 1
        for texts_batch in split_into_batches(texts, batch_size=batch_size):
            print(counter, len(texts_batch))
            # retry loop
            while True:
                try:
                    embeds_batch = self.get_embeddings(texts_batch)  # API CALL
                    embeddings += embeds_batch
                    break  # exit the retry loop and go to the next batch
                except openai.error.RateLimitError as err:
                    print(f"... Rate limit reached. Sleeping for {sleep_seconds} seconds.")
                    sleep(sleep_seconds)
                    # retry the same batch
                #except openai.error.InvalidRequestError as err:
                #    print("INVALID REQUEST", err)
            counter += 1
        return embeddings

    def get_embeddings_in_dynamic_batches(self, texts, batch_char_limit=30_000, sleep_seconds=60):
        """High level wrapper to work around API limitations

            RateLimitError:
                Rate limit reached for [MODEL] in [ORG] on tokens per min.
                Limit: 1_000_000 tokens / min.

            AND

            InvalidRequestError:
                This model's maximum context length is 8191 tokens,
                however you requested X tokens (X in your prompt; 0 for the completion).
                Please reduce your prompt; or completion length.

            Params:

                batch_char_limit : Number of max characters to request per API call.
                                    Should be less than around 32_000 based on API docs.

                sleep : Wait for a minute before requesting the next batch

        """
        embeddings = []
        counter = 1
        for texts_batch in dynamic_batches(texts, batch_char_limit=batch_char_limit):
            print(counter, len(texts_batch))
            # retry loop
            while True:
                try:
                    embeds_batch = self.get_embeddings(texts_batch)  # API CALL
                    embeddings += embeds_batch
                    break  # exit the retry loop and go to the next batch
                except openai.error.RateLimitError as err:
                    print(f"... Rate limit reached. Sleeping for {sleep_seconds} seconds.")
                    sleep(sleep_seconds)
                    # retry the same batch
            counter += 1
        return embeddings








if __name__ == "__main__":

    from app import DATA_DIRPATH

    print("-----------------")
    print("TEXTS:")
    texts = [
        "Short and sweet",
        "Short short",
        "I like apples, but bananas are gross.",
        "This is a tweet about bananas",
        "Drink apple juice!",
    ]
    pprint(texts)

    #print("-----------------")
    #print("BATCHES:")
    #batches = list(split_into_batches(texts, batch_size=2))
    #pprint(batches)

    #print("-----------------")
    #print("DYNAMIC BATCHES:")
    #batches = dynamic_batches(texts, batch_char_limit=30)
    #pprint(batches)

    print("-----------------")
    print("EMBEDDINGS:")

    ai = OpenAIService()

    embeddings = ai.get_embeddings(texts)
    #embeddings = ai.get_embeddings_in_dynamic_batches(texts, batch_char_limit=15_000)
    #print(type(embeddings), len(embeddings))
    #print(len(embeddings[0])) #> 1536

    df = DataFrame({"text": texts, "openai_embeddings": embeddings})
    print(df)

    print("-----------------")
    # UNpacK EMBEdDINGS tO THEIR OWN COLUMNS
    embeds_df = DataFrame(df["openai_embeddings"].values.tolist())
    embeds_df.columns = [str(i) for i in range(0, len(embeds_df.columns))]
    embeds_df = df.drop(columns=["openai_embeddings"]).merge(embeds_df, left_index=True, right_index=True)
    print(embeds_df)

    print("-----------------")
    print("SAVING...")

    model_dirpath = os.path.join(DATA_DIRPATH, ai.model_id)
    os.makedirs(model_dirpath, exist_ok=True)

    embeddings_csv_filepath = os.path.join(model_dirpath, "example_openai_embeddings.csv")
    #embeddings_json_filepath = os.path.join(model_dirpath, "example_openai_embeddings.json")

    embeds_df.to_csv(embeddings_csv_filepath)
    #df.to_json(embeddings_json_filepath)
