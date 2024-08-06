import argparse
import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger

from data.medical_data_module import MedicalDataModule
from models.resnet_binary_classifier import ResNetBinaryClassifier

torch.set_float32_matmul_precision('medium')


def train(args):
    print(args)
    transforms = {

        "train": A.Compose([A.Resize(height=137, width=137), ToTensorV2()]),
        "validation": A.Compose(
            [
                A.Compose([A.Resize(height=137, width=137), ToTensorV2()]),
            ]
        ),
    }
    tb_logger = TensorBoardLogger(args["logger_path"], name="isic_2024_logs")

    trainer = pl.Trainer(accelerator="gpu",
                         devices=[0],
                         max_epochs=args["epochs"],
                         logger=tb_logger,
                         precision='16-mixed')

    model = ResNetBinaryClassifier()
    dm = MedicalDataModule(
        image_path=args["image_path"],
        metadata_path=args["metadata_path"],
        batch_size=args["batch_size"],
        split_ratio=args["split_ratio"],
        transforms=transforms
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="_raw_data/train-image.hdf5")
    parser.add_argument("--metadata_path", type=str, default="_raw_data/train-metadata.csv")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--split_ratio", type=int, default=0.8)
    parser.add_argument("--logger_path", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(vars(args))
