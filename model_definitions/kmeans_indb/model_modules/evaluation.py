
import matplotlib.pyplot as plt
import teradataml as tdml
from teradataml import *   
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

def evaluate(context: ModelContext, **kwargs):
  print("All done as there is not need to exaluate kmean")
