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
    print(np.sum(df['target']))
    df = df[df['tbp_lv_dnn_lesion_confidence']>70]
    print("SAMPLES WITH HIGHER CONFIDENCE THAN 70: {}".format(np.sum(df['target'])))
    print(df.shape)
    df = df[df['tbp_lv_dnn_lesion_confidence'] > 90]
    print("SAMPLES WITH HIGHER CONFIDENCE THAN 90: {}".format(np.sum(df['target'])))
    print(df.shape)
    df = df[df['tbp_lv_dnn_lesion_confidence'] > 95]
    print("SAMPLES WITH HIGHER CONFIDENCE THAN 95: {}".format(np.sum(df['target'])))
    print(df.shape)

