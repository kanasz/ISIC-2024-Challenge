from io import BytesIO
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from utils.core_functions import get_preprocessor, get_data

hdf5_file = None
#Very ugly temporary fix
hdf5_file_path_global = "_raw_data/train-image.hdf5"


def worker_init_fn(worker_id):
    global hdf5_file
    global hdf5_file_path_global
    hdf5_file = h5py.File(hdf5_file_path_global, 'r')


class MedicalDataset(Dataset):
    def __init__(self, metadata, transformations=None, hdf5_file_path=None, preprocessor=None):
        #self.images = images
        self.metadata = metadata
        self.transformations = transformations
        self.hdf5_file_path = hdf5_file_path
        self.preprocessor = preprocessor
        #self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        if self.preprocessor:
            self.metadata_processed = self.preprocessor.transform(self.metadata)

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, idx):
        global hdf5_file

        #with h5py.File(self.hdf5_file_path, 'r') as f:
        #df_selected = self.metadata[self.metadata['isic_id']].iloc[idx]
        df_selected = self.metadata.iloc[idx]
        #df_selected = self.metadata[self.metadata['isic_id']==idx]
        label = df_selected['target']
        isic_id = df_selected['isic_id']

        image = Image.open(BytesIO(hdf5_file[isic_id][()]))

        #image = image.convert('RGB')
        image = np.array(image)

        if self.transformations:
            image = self.transformations(image=image)['image']

        if hasattr(self, 'metadata_processed'):
            self.selected_row = torch.tensor(self.metadata_processed[[idx]], dtype=torch.float32)
            return (image, self.selected_row), label
        else:
            return image, label


class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, image_path, metadata_path, batch_size=32, split_ratio=0.8, subset_ratio=0.5, transforms=None,
                 training_minority_oversampling_ceoff=5, lesion_confidence_threshold = 95):
        super().__init__()
        global hdf5_file_path_global
        hdf5_file_path_global = image_path

        self.lesion_confidence_threshold = lesion_confidence_threshold
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.subset_ratio = subset_ratio  # Ratio of training data to use in each epoch
        self.transforms = transforms
        self.train_indices = None
        self.val_indices = None
        self.minority_indices = None
        self.majority_indices = None
        self.df_metadata = None
        self.pos_weight = torch.tensor([100], device="cuda")
        self.weights = torch.tensor([1,100], device="cuda")
        self.cls_num_list = torch.tensor([], device="cuda")
        self.training_minority_oversampling_ceoff = training_minority_oversampling_ceoff


    def setup(self, stage=None):
        self.df_metadata = get_data(self.metadata_path, lesion_confidence_threshold=self.lesion_confidence_threshold)

        patient_ids = self.df_metadata['patient_id'].values
        labels = self.df_metadata['target'].values

        # Group by unique patients
        unique_patient_ids = np.unique(patient_ids)

        # Temporary dataframe for stratified splitting by patient
        temp_df = pd.DataFrame({'patient_id': unique_patient_ids})
        temp_df['label'] = temp_df['patient_id'].map(lambda pid: labels[np.where(patient_ids == pid)[0][0]])

        # Perform stratified split on patients, not individual samples
        train_patients, val_patients = train_test_split(
            temp_df['patient_id'],
            test_size=1 - self.split_ratio,
            stratify=temp_df['label'],
            random_state=42
        )

        # Map back to indices in the original dataset
        train_indices_all = self.df_metadata[self.df_metadata['patient_id'].isin(train_patients)].index.tolist()

        self.preprocessor, features_processed = get_preprocessor(self.df_metadata[(self.df_metadata.index.isin(train_indices_all))])
        self.processed_features_shape = features_processed.shape
        self.val_indices = self.df_metadata[self.df_metadata['patient_id'].isin(val_patients)].index.tolist()

        # Identify minority and majority class samples
        minority_class = self.df_metadata['target'].value_counts().idxmin()
        self.minority_indices = self.df_metadata[(self.df_metadata.index.isin(train_indices_all)) & (
                self.df_metadata['target'] == minority_class)].index.tolist()


        minority_count = len(self.minority_indices)
        majority_count = len(train_indices_all) - len(self.minority_indices)

        self.minority_indices = self.minority_indices * self.training_minority_oversampling_ceoff

        self.majority_indices = self.df_metadata[(self.df_metadata.index.isin(train_indices_all)) & (
                self.df_metadata['target'] != minority_class)].index.tolist()

        self.minority_val_indices = self.df_metadata[(self.df_metadata.index.isin(self.val_indices)) & (
                self.df_metadata['target'] == minority_class)].index.tolist()

        self.majority_val_indices = self.df_metadata[(self.df_metadata.index.isin(self.val_indices)) & (
                self.df_metadata['target'] != minority_class)].index.tolist()

        # Debugging: Print information about indices
        print(f"Total training indices: {len(train_indices_all)}")
        print(f"Minority training class indices: {len(self.minority_indices)}")
        print(f"Majority training class indices: {len(self.majority_indices)}")
        print(f"Total validation indices: {len(self.val_indices)}")
        print(f"Minority validation indices: {len(self.minority_val_indices)}")

        #OPRAVIT - len z minority pred oversamplovanim
        self.pos_weight = torch.tensor(
            [(int(len(self.majority_indices))) / minority_count], device="cuda")

        self.weights = torch.tensor(
            [(majority_count + minority_count )/ majority_count, (majority_count + minority_count )/ minority_count], device="cuda")
        #self.pos_weight = torch.tensor([(int(len(self.majority_indices) * self.subset_ratio)) / len(self.minority_indices)], device="cuda")
        self.cls_num_list = torch.tensor([len(self.majority_indices) * self.subset_ratio, len(self.minority_indices)],
                                         device="cuda")

        self.val_dataset = Subset(MedicalDataset(self.df_metadata, transformations=self.transforms['validation'],
                                                hdf5_file_path=self.image_path, preprocessor=self.preprocessor),
                                  self.val_indices)

        self.train_dataset = None
        self.update_train_dataset()

    def update_train_dataset(self):
        f = h5py.File(self.image_path, 'r')
        # Randomly sample a subset of the majority class samples
        n_samples_majority = int(len(self.majority_indices) * self.subset_ratio)
        sampled_majority_indices = np.random.choice(self.majority_indices, size=n_samples_majority, replace=False)

        # Combine minority samples with the resampled majority samples
        sampled_train_indices = self.minority_indices + sampled_majority_indices.tolist()


        # Debugging: Print sampled indices
        print(f"Sampled majority class indices: {len(sampled_majority_indices)}")
        print(f"Total sampled training indices: {len(sampled_train_indices)}")
        print(f"Total minority training indices: {len(self.minority_indices)}")
        if not sampled_train_indices:
            raise ValueError("Sampled training indices are empty. Check your sampling logic.")

        #self.pos_weight = torch.tensor([len(sampled_majority_indices) / len(self.minority_indices)], device="cuda")
        # Update the train dataset with the new subset
        self.train_dataset = Subset(MedicalDataset(self.df_metadata, transformations=self.transforms['train'],
                                                   hdf5_file_path=self.image_path, preprocessor=self.preprocessor), sampled_train_indices)

    def train_dataloader(self):
        self.update_train_dataset()  # Update dataset at each epoch
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4 , persistent_workers=True,
                          worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, persistent_workers=True,
                          worker_init_fn=worker_init_fn)
