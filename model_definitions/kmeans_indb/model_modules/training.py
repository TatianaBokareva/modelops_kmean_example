import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import aoa
from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

import teradataml as tdml
from teradataml import *


# Configure Teradata Vantage analytics functions to a specific installation location
configure.val_install_location = 'val'


def train(context: ModelContext, **kwargs):
    """
    The main training function for building a model

    Parameters:
    - context (ModelContext): Contextual information and metadata about the model and training environment.
    
    Keyword Arguments:
    - Various hyperparameters and configurations can be passed used directly from the context.
    
    Returns:
    - None
    """
    aoa_create_context()  # Initialize the AOA context for the model training session

    # Extract dataset information from the context
    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names
    entity_key = context.dataset_info.entity_key
    category_features = []
    
    if type(target_name) == list:
        target_name = target_name[0]
        
    if type(entity_key) == list:
        entity_key = entity_key[0]

    # Load the training dataset from Teradata as a DataFrame
    print("ontext.dataset_info.sql")
    
    train_df = DataFrame.from_query(context.dataset_info.sql)
    print('Feature names:', feature_names)
    print(train_df.head())

    # Record training statistics
    record_training_stats(
        train_df,
        features    = [c for c in feature_names if c not in [target_name] + category_features],
        targets     = [target_name],
        categorical = category_features + [target_name],
        context     = context
    )

    # Train a kmeans model
    print("Kmeans training")
    KMeans_out = KMeans(id_column="id"
                          ,target_columns=['sepal_length','sepal_width','petal_length','petal_width']
                          ,data=train_df
                         ,num_clusters = 3)
    
    print("Printing centorids")
    
    df_model = KMeans_out.result.to_pandas()
    x = list(df_model[df_model['sepal_length'].notnull()]["sepal_length"])
    y = list(df_model[df_model['petal_length'].notnull()]["petal_length"])
    z = list(df_model[df_model['sepal_width'].notnull()]["sepal_width"])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('TotalQuantity')
    ax.set_ylabel('TotalPrice')
    ax.set_zlabel(r'TotalItems')
    plt.title('Centroids')

   savefig("centroids.png", dpi=500)
    
    # Save the trained model object to SQL: The table is called model_long_id_of the model
    KMeans_out.result.to_sql(f"model_{context.model_version}", if_exists="replace")
    
    print("Saved trained model", f"model_{context.model_version}")
    
    copy_to_sql(df = KMeans_out.result, table_name = 'k_means_model', if_exists='replace')
    print("All done!")
