from io import BytesIO
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class MedicalDataset(Dataset):
    def __init__(self, images, metadata, transformations=None):
        self.images = images
        self.metadata = metadata
        self.transformations = transformations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        df_selected = self.metadata.iloc[[idx]]
        label = df_selected['target'].values[0]
        metadata = df_selected
        metadata = metadata.drop(columns=['target'])
        isic_id = metadata['isic_id'].values[0]
        image = Image.open(BytesIO(self.images[isic_id][()]))
        image = np.array(image)

        if self.transformations:
            transformed = self.transformations(image=image)  # Apply transformation
            image = transformed['image']
        image = image / 255

        return image, label


class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, image_path, metadata_path, batch_size=32, split_ratio=0.8, transforms=None):
        super().__init__()
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.transforms = transforms

    def setup(self, stage=None):
        df_metadata = pd.read_csv(self.metadata_path, index_col=False)
        f = h5py.File(self.image_path, 'r')
        patient_ids = df_metadata['patient_id']
        unique_patients = list(set(patient_ids))
        train_patients, val_patients = train_test_split(unique_patients, train_size=self.split_ratio, random_state=42)

        train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
        val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]

        self.train_dataset = Subset(MedicalDataset(f, df_metadata, transformations=self.transforms['train']),
                                    train_indices)
        self.val_dataset = Subset(MedicalDataset(f, df_metadata, transformations=self.transforms['validation']),
                                  val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
