


from app.model_storage import StorageService, BUCKET_NAME


def test_storage_service():

    storage = StorageService()

    # the bucket exists:
    assert storage.bucket.name == BUCKET_NAME

    breakpoint()
