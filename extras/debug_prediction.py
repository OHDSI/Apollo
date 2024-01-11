import pandas as pd
import pyarrow.parquet as pq
import os

from training import train_model

folder = "/Users/schuemie/Data/DebugCovariateData/person_sequence"
full = pd.read_parquet(folder)

folder = "/Users/schuemie/Data/DebugCovariateData/person"
full = pd.read_parquet(folder)

train_model.main(["extras/my_model_prediction.yaml", "/Users/schuemie/Data/DebugCovariateData/prediction.csv"])