import torch
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py
import io
import numpy as np
import pandas as pd


class MedicalDataset(Dataset):
    def __init__(self, images, metadata, labels):
        self.images = images
        self.metadata = metadata
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        metadata = self.metadata[idx]
        image = torch.tensor(image).permute(2, 0, 1).float()
        return image, metadata, label


class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, data_path, metadata_path, batch_size=32, split_ratio=0.8):
        super().__init__()
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        images = []
        metadata_list = []

        df = pd.read_csv(metadata_path, index_col=False)
        f = h5py.File(data_path, 'r')
        patient_ids = df['patient_id']
        isic_ids = df['isic_id']
        labels = df['target']

        for index, key in enumerate(f.keys()):
            if index % 1000 == 0:
                print(index)
            is_true = isic_ids[index] == key
            if is_true == False:
                print("ERROR IN DATA: {}, {}".format(isic_ids[index], key))
            dset = f[key]
            img_plt = Image.open(io.BytesIO(np.array(dset)))
            img_array = np.array(img_plt)
            images.append(img_array)
            metadata_list.append(df.loc[df['isic_id'] == key].to_dict(orient='records')[0])

        unique_patients = list(set(patient_ids))
        train_patients, val_patients = train_test_split(unique_patients, train_size=self.split_ratio, random_state=42)

        train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
        val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]

        self.train_dataset = Subset(MedicalDataset(images, metadata_list, labels), train_indices)
        self.val_dataset = Subset(MedicalDataset(images, metadata_list, labels), val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_image_paths = []
        test_labels = []
        test_dataset = MedicalDataset(test_image_paths, test_labels)
        return DataLoader(test_dataset, batch_size=self.batch_size)


data_path = '../_raw_data/train-image.hdf5'
metadata_path = '../_raw_data/train-metadata.csv'
data_module = MedicalDataModule(data_path, metadata_path, batch_size=64)
data_module.setup()


class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(3 * 224 * 224, 2)  # Adjust input dimensions and output dimensions as needed

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = MyModel()
trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, datamodule=data_module)
