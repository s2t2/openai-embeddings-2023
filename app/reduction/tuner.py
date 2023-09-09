
import os
from abc import ABC, abstractmethod
from pandas import DataFrame
import plotly.express as px

from app import RESULTS_DIRPATH
from app.dataset import Dataset
from app.reduction.pipeline import ReductionPipeline, REDUCER_TYPE #, FIG_SAVE, FIG_SHOW

MAX_COMPONENTS = os.getenv("MAX_COMPONENTS")


class ReductionTuner(ABC):

    def __init__(self, reducer_type, ds=None, results_dirpath=RESULTS_DIRPATH, max_components=MAX_COMPONENTS):
        self.ds = ds or Dataset()
        #self.df = self.ds.df
        #self.label_cols = self.ds.label_cols
        self.feature_names = self.ds.feature_cols # self.df.drop(columns=self.label_cols).columns.tolist()

        self.reducer_type = reducer_type
        if max_components:
            max_components = int(max_components)
        self.max_components = max_components

        self.results_dirpath = results_dirpath
        #os.makedirs(self.results_dirpath, exist_ok=True)

        self.results = None
        self.results_df = None

    #def perform(self):
    #    raise NotImplementedError("Define this method in the child class. It should return a results_df with a row per component, along with any available metrics.")

    def perform(self):
        self.results = []

        # if we have lots of columns / features, we might want to abbreviate the search space and override with a max value, otherwise search over all available features
        max_components = self.max_components or len(self.feature_cols)
        # get the explained variance for each n up to the max number of components to search over
        for n_components in range(1, max_components+1):
            # we need to use PCA specifically because unlike other methods it gives us the explainability metrics
            pipeline = ReductionPipeline(ds=self.ds, reducer_type=self.reducer_type, n_components=n_components)
            pipeline.perform()
            self.collect_result(pipeline, n_components)

        self.results_df = DataFrame(self.results)
        self.print_results()

    @abstractmethod
    def collect_result(self, pipeline, n_components):
        #raise NotImplementedError
        pass

    @abstractmethod
    def print_results(self):
        #raise NotImplementedError
        pass


#if __name__ == "__main__":
#
#
#    from app.dataset import Dataset
#
#    ds = Dataset()
#    tuner = ReductionTuner(df=ds.df, label_cols=ds.label_cols)
#    #> Can't instantiate abstract class ReductionTuner with abstract methods collect_result, print_results
#    breakpoint()
#
