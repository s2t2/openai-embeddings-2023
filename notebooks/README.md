





# Notebooks and Code

This section provides a walk-through of the methods, with working code for reference. The process starts with three Python notebooks and follows up with Python scripts in a larger code repository.

The three notebooks collectively perform Methodology Steps 1 and 2 (obtaining embeddings for a sample of users).

> Note: We could have used one notebook, but the second and third notebooks clean up the embeddings so they are in an easier to use format, and merge the embeddings with some additional botometer scores we forgot to include originally.

The third notebook produces a CSV file that is then used in the code repository to perform the remaining methodology steps (dimensionality reduction, classification, etc).


## Notebook 1: Botometer User Sample and OpenAI Embeddings

Queries Impeachment 2020 dataset for users for which we have botometer scores 7,566 users), and also fetches their tweets (50 tweets maximum per user). Concatenates all of a user's tweets into a single string.

Obtains embeddings for each user's concatenated tweet texts, via the OpenAI API. Obtains a single vector of 1536 embeddings for each user, representing all of their tweets. Uses the "text-embedding-ada-002" model. In practice, "This model's maximum context length is 8,191 tokens". Since each token is around four characters, the model's context limit is around ~32,00 characters. So each user must have less characters than this to obtain embeddings.
In practice, we requested embeddings in batches, using a max tweet character length per batch of 15,000. This just controls the number of users to request in each batch. Most users had total tweet character lengths below 10,000. In practice, we encountered API rate limits (1M tokens per minute), so we implemented a sleep for one minute whenever the limit was reached.

Saves a single file of the embeddings (one column for profile embeddings list, and one column for tweets embeddings lists) as "botometer_sample_openai_embeddings_20230704".



## Notebook 2: Embeddings Data Export

> Note: Since the file previously exported has the embeddings as a JSON string, which are hard to use. And it contains both profile and tweet embeddings, but we want to focus on just tweet embeddings, so we split the datasets in two and improve formatting.

Takes the file "botometer_sample_openai_embeddings_20230704", unpacks the embeddings into their own columns, and splits into two datasets: one for tweets (botometer_sample_openai_tweet_embeddings_20230704"), and one for profiles ("botometer_sample_openai_profile_embeddings_20230704").

> Note: only the tweets dataset is used moving forward.

## Notebook 3: Merging Remaining Botometer Scores

Takes the file "botometer_sample_openai_tweet_embeddings_20230704", merges in remaining botometer scores not originally included, and creates the file: "botometer_sample_openai_tweet_embeddings_20230724" for use by the code repository.
