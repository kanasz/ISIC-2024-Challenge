import h5py
import numpy as np
import io

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("MAIN")

    df = pd.read_csv("_raw_data/train-metadata.csv")
    print(df.shape)
    #print(df.columns)

    #df_filtered = df[pd.isna(df['iddx_2'])==False]
    #print(df_filtered.shape)
    #df_filtered = df[pd.isna(df['iddx_3']) == False]
    #print(df_filtered.shape)
    print(np.sum(df['target']))
    df = df[df['tbp_lv_dnn_lesion_confidence']>95]
    print(np.sum(df['target']))
    print(df.shape)
    print(df[pd.isna(df['iddx_2'])==False])

    print(df)

