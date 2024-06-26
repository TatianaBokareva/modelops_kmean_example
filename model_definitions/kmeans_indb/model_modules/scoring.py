import teradataml as tdml
from teradataml import *

from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def score(context: ModelContext, **kwargs):
    """
    Execute scoring on a provided dataset using a pre-trained model 
    Parameters:
    - context (ModelContext): Contextual information including metadata about model version,
                              dataset details, and job configuration.

    Keyword Arguments:
    - kwargs: Additional keyword arguments (not used here).

    Returns:
    - None
    """
    aoa_create_context()  # Initialize AOA context for scoring session

    # Load the model from the database
    model = DataFrame(f"model_${context.model_version}")
    Kmean_out = DataFrame(in_schema("demo_user","k_means_model"))
    
    tdf = DataFrame.from_query(context.dataset_info.sql)
    print("context.dataset_info.sql")

    KMeansPredict_out = KMeansPredict(object=KMeans_out.result,
                                      data=tdf)
    
    print("Saved predictions in Teradata")
    copy_to_sql(df = KMeansPredict_out.result, table_name = 'kmean_score', if_exists='replace')
