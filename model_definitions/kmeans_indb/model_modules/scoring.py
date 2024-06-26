import teradataml as tdml
from teradataml import *

from teradataml import (
    copy_to_sql,
    DataFrame,
    KMeansPredict
)

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
    print("getting model: ",f"model_{context.model_version}")
    Kmean_out = DataFrame(f"model_{context.model_version}")
    
    print("Getting Scoring data:", context.dataset_info.sql)    
    tdf = DataFrame.from_query(context.dataset_info.sql)
  
    print("Scoring")
    KMeansPredict_out = KMeansPredict(object=Kmean_out,data=tdf).result


    # Retrieve target, and entity key names from the model context. Note: order columns to match the expected schema in the database
    tbl_out = KMeansPredict_out.assign(drop_columns=True
                             # Add job_id to track the execution
                            ,job_id = context.job_id 
                             # Set entity key from the features_pdf
                            ,CustomerID = KMeansPredict_out.id.cast(type_=BIGINT)
                             # rename td_clusterid_kmeans to target_names
                            ,td_clusterid_kmeans = KMeansPredict_out.td_clusterid_kmeans.cast(type_=INTEGER) 
                            # Add an empty json_report column for compatibility with the expected table schema
                            ,json_report= ""
                            ).select(["job_id", "CustomerID", "td_clusterid_kmeans","json_report"])  

    
    sc_tb = DataFrame(in_schema(context.dataset_info.predictions_database, context.dataset_info.predictions_table))

    print("Finished Scoring")

    # Append the results to the specified prediction table in Teradata
    print("Saved predictions in Teradata")
    copy_to_sql(
        df=tbl_out,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
         if_exists="append"
    )
    print("All done")
