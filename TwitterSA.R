import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("./Users/gaddamnitish/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols)

df.head()