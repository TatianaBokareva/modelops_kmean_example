
import matplotlib.pyplot as plt
import seaborn as sns

import teradataml as tdml
from teradataml import *   
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()
    #load the model
    #KMeans_out = DataFrame(in_schema("demo_user","k_means_model"))
    
    print(f"Loading model from table model_{context.model_version}")
    KMeans_out = DataFrame(f"model_{context.model_version}")
    print(context.dataset_info.sql)
    
    # Load the test data set
    tdf = DataFrame.from_query(context.dataset_info.sql)
    #tdf = DataFrame(in_schema("demo_user","iris")) 
    KMeansPredict_out = KMeansPredict(object=KMeans_out,data=tdf)

    tdf1 = tdf.join(other = KMeansPredict_out.result, on = ["id=id"], how = "inner", lprefix = "t1")
    
    #Exclude unwanted colums
    obj = Antiselect(data=tdf1,exclude=['t1_id'])
    df_kmeans_scored = obj.result.to_pandas()
    
    print("generating picture")
    x = list(df_kmeans_scored[df_kmeans_scored['sepal_length'].notnull()]["sepal_length"])
    y = list(df_kmeans_scored[df_kmeans_scored['petal_length'].notnull()]["petal_length"])
    z = list(df_kmeans_scored[df_kmeans_scored['sepal_width'].notnull()]["sepal_width"])
    col = list(df_kmeans_scored[df_kmeans_scored['td_clusterid_kmeans'].notnull()]["td_clusterid_kmeans"])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=col)
    ax.set_xlabel('sepal_length')
    ax.set_ylabel('petal_length')
    ax.set_zlabel('sepal_width')
    plt.title('Scored')
    # Save the plot to a file
    print("Print path to file: ",context.artifact_output_path)
    plt.gcf().savefig(f"{context.artifact_output_path}/3_d_plot_points.png", dpi=500)
    plt.clf()  # Clear the plot to free memory
    
    ## Create classification matrix
    print("Creating misscalssification matrix")
    df_cnt = df_kmeans_scored.groupby(['species','td_clusterid_kmeans']).size()
    df_c = df_cnt.to_frame().reset_index()
    df_c["cnt"] = df_c[0]
    df_c = df_c.drop(columns=[0])

    #Combine the keys and species into one column
    df_c["keys"] = df_c.apply(lambda x: '_'.join(x[["species","td_clusterid_kmeans"]].astype(str).values), axis=1)

    # Create dictionary 
    evaluation = {}

    def add_values(row):
        evaluation[row["keys"]] = row['cnt']
    
    # Apply the user-defined function to every row
    df_c.apply(add_values, axis=1)
    print(evaluation)

    
    ## ADD CODE TO SAFE PLOT
    print("printing barcplots for seaborn")
    fig = sns.barplot(x=df_c.td_clusterid_kmeans, y=df_c.cnt, hue = df_c.species)
    fig.savefig(f"{context.artifact_output_path}/barplots.png", dpi=500)
    
    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)
    
    print("All done")
