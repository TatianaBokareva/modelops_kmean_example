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
    print("getting model: ",f"model_{context.model_version}")
    Kmean_out = DataFrame(f"model_{context.model_version}")
    
    print("Getting Scoring data:", context.dataset_info.sql)    
    tdf = DataFrame.from_query(context.dataset_info.sql)
  
    print("Scoring")
    KMeansPredict_out = KMeansPredict(object=Kmean_out,data=tdf).result
    print(KMeansPredict_out)
    # Convert predictions to pandas DataFrame, adjust the column name to target_name, and ensure the type is integer
    # Retrieve target, and entity key names from the model context
    target_name = context.dataset_info.target_names
    entity_key = context.dataset_info.entity_key
    predictions_pdf = KMeansPredict_out.to_pandas(all_rows=True).rename(columns={"td_clusterid_kmeans": target_name}).astype({target_name: int})

    print("Finished Scoring")
    # Prepare the predictions DataFrame for database insertion
    predictions_pdf["job_id"] = context.job_id  # Add job_id to track the execution
    predictions_pdf[entity_key] = entity_key  # Set entity key from the features_pdf
    predictions_pdf["json_report"] = ""  # Add an empty json_report column for compatibility with the expected table schema

    # Reorder columns to match the expected schema in the database
    predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    # Append the results to the specified prediction table in Teradata
    copy_to_sql(
        df=predictions_pdf,
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
         if_exists="append"
    )

    print("Saved predictions in Teradata")
    # copy_to_sql(df = KMeansPredict_out.result, table_name = 'kmean_score', if_exists='replace')
    print("All done")
