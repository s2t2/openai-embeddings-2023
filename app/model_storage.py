import os
import joblib
from functools import cached_property

from google.cloud import storage as gcs
from dotenv import load_dotenv

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # implicit check by google.cloud for env var

PROJECT_ID = os.getenv("GOOGLE_PROJECT_NAME") # "my-project"
BUCKET_NAME = os.getenv("BUCKET_NAME") # "my-bucket" needs to be globally unique!


class ModelStorage:

    def __init__(self, model_dirname:str, project_id=PROJECT_ID, bucket_name=BUCKET_NAME):
        """ Params model_dirname like 'my-model' """
        self.project_id = project_id
        self.bucket_name = bucket_name
        print("PROJECT ID:", self.project_id)
        print("BUCKET NAME:", self.bucket_name)

        self.model_dirname = model_dirname
        self.model_filepath = os.path.join(self.model_dirname, "model.joblib")
        self.local_model_filepath = self.model_filepath
        self.hosted_model_filepath = self.model_filepath
        print("MODEL DIRNAME:", self.model_dirname)


    @property
    def client(self):
        return gcs.Client(project=self.project_id)

    @cached_property
    def buckets(self):
        return list(self.client.list_buckets())

    def find_or_create_bucket(self):
        for bucket in self.buckets:
            if bucket.name == self.bucket_name:
                print("USING EXISTING BUCKET...")
                return bucket

        print(f"CREATING BUCKET...")
        return self.client.create_bucket(self.bucket_name)

    @cached_property
    def bucket(self):
        return self.find_or_create_bucket()

    @property
    def model_blob(self):
        return self.bucket.blob(self.hosted_model_filepath)

    def upload_model(self, model):
        print("DUMPING MODEL...")
        os.makedirs(self.model_dirname, exist_ok=True)
        joblib.dump(model, self.local_model_filepath)
        print("UPLOADING MODEL...")
        self.upload_model_from_file()

    def upload_model_from_file(self):
        self.model_blob.upload_from_filename(self.local_model_filepath)

    def download_model(self):
        print("DOWNLOADING MODEL...")
        with self.model_blob.open(mode="rb") as file:
            return joblib.load(file)


if __name__ == "__main__":


    storage = ModelStorage("example-model")

    print("---------------------")
    for bucket in storage.buckets:
        print(bucket)

    print("---------------------")
    print(storage.bucket)
