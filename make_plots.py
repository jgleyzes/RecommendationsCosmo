import pandas as pd
import os

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

dfADS = pd.read_csv(os.path.join(THIS_PATH,'data/inspireDB.csv'))
