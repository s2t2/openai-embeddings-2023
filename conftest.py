#import os
#from pytest import fixture
#
#from pandas import read_csv
#
#from app import DATA_DIRPATH
#
#CI_ENV = bool(os.getenv("CI")=="true")
#
#TWEETS_CSV_FILEPATH = os.path.join(DATA_DIRPATH, "text-embedding-ada-002")
#
#@fixture(scope="module")
#def tweets_df():
#    return read_csv(TWEETS_CSV_FILEPATH)
#
