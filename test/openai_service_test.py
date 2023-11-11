
import os

from pytest import fixture
from pandas import read_csv

from app import DATA_DIRPATH
from app.openai_service import split_into_batches, dynamic_batches


def test_batchmakers():

    texts = [
        "Short and sweet",
        "Short short",
        "I like apples, but bananas are gross.",
        "This is a tweet about bananas",
        "Drink apple juice!",
    ]

    batches = list(split_into_batches(texts, batch_size=2))
    assert batches == [
        ['Short and sweet', 'Short short'], # 2
        ['I like apples, but bananas are gross.', 'This is a tweet about bananas'], # 2
        ['Drink apple juice!'] # remainder
    ]

    batches = dynamic_batches(texts, batch_char_limit=30)
    assert batches == [
        ['Short and sweet', 'Short short'],
        ['I like apples, but bananas ar'],
        ['This is a tweet about bananas'],
        ['Drink apple juice!']
 ]




#@fixture(scope="module")
#def example_embeddings_df();


def test_load_embeddings():

    example_embeddings_csv_filepath = os.path.join(os.path.dirname(__file__), "data", "text-embedding-ada-002", "example_openai_embeddings.csv")
    print(os.path.isfile(example_embeddings_csv_filepath))
    embeds_df = read_csv(example_embeddings_csv_filepath)
    embeds_df.drop(columns=["Unnamed: 0"], inplace=True)

    assert "text" in embeds_df.columns
    assert embeds_df.drop(columns=["text"]).shape == (5, 1536)
