
#import os
#from abc import ABC
#from pandas import DataFrame
#import plotly.express as px
#
#from app import RESULTS_DIRPATH
#
#MAX_COMPONENTS = os.getenv("MAX_COMPONENTS")
#
#
#class ReductionTuner(ABC):
#
#    def __init__(self, df, label_cols=[], reducer_type="PCA", results_dirpath=RESULTS_DIRPATH, max_components=MAX_COMPONENTS):
#        self.df = df
#
#        self.label_cols = label_cols
#        self.feature_names = self.df.drop(columns=self.label_cols).columns.tolist()
#
#        if max_components:
#            max_components = int(max_components)
#        self.max_components = max_components
#
#        self.results_dirpath = results_dirpath
#        #os.makedirs(self.results_dirpath, exist_ok=True)
#
#        self.results = None
#        self.results_df = None
#
