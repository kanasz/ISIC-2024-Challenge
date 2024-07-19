import h5py
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f = h5py.File('_raw_data/train-image.hdf5', 'r')
    images = {}
    for key in list(f.keys())[0:16]:
        dset = f[key]
        img_plt = Image.open(io.BytesIO(np.array(dset)))
        img_array = np.array(img_plt)
        images[key] = img_array

    images_label = list(images.keys())
    images_array = list(images.values())

    plt.figure(figsize=(16, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(images_array[i])
        plt.title(images_label[i])
        plt.axis('off')
    plt.show()

