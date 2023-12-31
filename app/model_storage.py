import os
import joblib
from functools import cached_property

from google.cloud import storage as gcs
from dotenv import load_dotenv

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") # implicit check by google.cloud for env var

#PROJECT_ID = os.getenv("GOOGLE_PROJECT_NAME") # "my-project"
BUCKET_NAME = os.getenv("BUCKET_NAME") # "my-bucket" needs to be globally unique!


class StorageService:
    def __init__(self, bucket_name=BUCKET_NAME): # project_id=PROJECT_ID
        #self.project_id = project_id
        self.bucket_name = bucket_name
        print("-----------------------")
        print("CLOUD STORAGE SERVICE...")
        #print("PROJECT ID:", self.project_id)
        print("BUCKET NAME:", self.bucket_name)


    @property
    def client(self):
        return gcs.Client() # project=self.project_id

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

    def list_model_blobs(self):
        # https://cloud.google.com/storage/docs/json_api/v1/objects/list#list-object-glob
        return self.client.list_blobs(self.bucket_name, match_glob="**/model.joblib")





class ModelStorage(StorageService):

    def __init__(self, local_dirpath:str, bucket_name=BUCKET_NAME, storage_dirpath=None): # project_id=PROJECT_ID
        """ Params local_dirpath, assumed to be somewhere in the results dir"""
        super().__init__(bucket_name=bucket_name) # project_id=project_id

        self.local_dirpath = local_dirpath
        print("RESULTS DIR:", self.local_dirpath)

        self.storage_dirpath = storage_dirpath or self.local_dirpath.split("..")[-1] #> "/results/onwards/" # TODO: this leaves an initial slash, which may create a redundant "/" directory on cloud storage, so consider removing initial slash if possible in the future (oops already saved all the models there :-D)
        print("STORAGE DIR:", self.storage_dirpath)

        self.model_filename = "model.joblib" # needs to be called 'model.joblib' specifically, for hosting from cloud storage on Google Vertex AI
        self.local_model_filepath = os.path.join(self.local_dirpath, self.model_filename)
        self.hosted_model_filepath =  os.path.join(self.storage_dirpath, self.model_filename)


    @property
    def model_blob(self):
        return self.bucket.blob(self.hosted_model_filepath)

    def save_model(self, model):
        print("SAVING MODEL (LOCAL)...")
        os.makedirs(self.local_dirpath, exist_ok=True)
        joblib.dump(model, self.local_model_filepath)

    def upload_model_from_file(self):
        print("UPLOADING MODEL...")
        self.model_blob.upload_from_filename(self.local_model_filepath)

    def save_and_upload_model(self, model):
        self.save_model(model)
        self.upload_model_from_file()

    def download_model(self):
        print("DOWNLOADING MODEL...")
        with self.model_blob.open(mode="rb") as file:
            return joblib.load(file)




if __name__ == "__main__":


    storage = StorageService()

    print("---------------------")
    for bucket in storage.buckets:
        print(bucket)

    print("---------------------")
    print(storage.bucket)

    print("---------------------")

    blobs = list(storage.bucket.list_blobs())
    for blob in blobs:
        print("...", blob)

    breakpoint()
