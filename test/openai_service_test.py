from pytest import fixture

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

    embeddings_csv_filepath =
