

#import os
#from app import RESULTS_DIRPATH
#CLUSTERING_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "clustering")



# TODO: do more research about which clustering metrics to use
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
# https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
#
# UNSUPERVISED:
#
#   silhouette_score: The best value is 1 and the worst value is -1.
#       ... Values near 0 indicate overlapping clusters.
#
#   calinski_harabasz_score: higher score relates to a model with better defined clusters.
#
#   davies_bouldin_score: The minimum score is zero, with lower values indicating better clustering.
#
#
# SUPERVISED:
#   adjusted_rand_score: Perfect labeling is scored 1.0.
#       ... Poorly agreeing labels (e.g. independent labelings) have lower scores,
#       ... and for the adjusted Rand index the score will be negative or close to zero.
#
