
import matplotlib.pyplot as plt
    


def evaluate(context: ModelContext, **kwargs):
    """
    Performs evaluation of a model using test data, plot points and created bar chars

    Parameters:
    - context (ModelContext): The model context object containing configuration and paths.

    Keyword Arguments:
    - kwargs: Miscellaneous keyword arguments.

    Returns:
    - None
    """
    aoa_create_context()  # Set up the AOA context for evaluation
    scored_tdf = DataFrame(in_schema("demo_user","kmean_score"))
    train_tdf =  DataFrame(in_schema("demo_user","kmean_score"))
   
    #tdf1 = tdf.join(other = KMeansPredict_out.result
     #       , on = ["id=id"], how = "inner", lprefix = "t1")
    #obj = Antiselect(data=tdf1,
    #                 exclude=['t1_id'])
