import argparse

import albumentations as A
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.loggers import TensorBoardLogger

from data.medical_data_module import MedicalDataModule
from loss_functions.ldam_loss import LDAMLoss
from models.efficientnet_binary_classifier import EfficientNetBinaryClassifier
from models.resnet_binary_classifier import ResNetBinaryClassifier

torch.set_float32_matmul_precision('medium')


def train(args):
    print(args)



    transforms = {
        "train": A.Compose([
            A.Resize(height=224, width=224),
            #A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ToTensorV2()
        ]),
        "validation": A.Compose([
            A.Resize(height=224, width=224),
            #A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            ToTensorV2(),
        ]),
    }

    dm = MedicalDataModule(
        image_path=args["image_path"],
        metadata_path=args["metadata_path"],
        batch_size=args["batch_size"],
        split_ratio=args["split_ratio"],
        transforms=transforms,
        subset_ratio=args["subset_ratio"]
    )
    dm.setup()
    pos_weight = dm.pos_weight
    weights = dm.weights

    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    w1 =dm.cls_num_list[1] / ( dm.cls_num_list[0] + dm.cls_num_list[1])
    w2 = dm.cls_num_list[0] / ( dm.cls_num_list[0] + dm.cls_num_list[1])
    criterion = LDAMLoss(dm.cls_num_list, weight=torch.tensor([w1, w2],device="cuda"))

    model = ResNetBinaryClassifier(learning_rate=args["lr"], criterion=criterion)
    # model = EfficientNetBinaryClassifier(learning_rate=args["lr"], criterion=criterion)

    tb_logger = TensorBoardLogger(args["logger_path"], name="{}_{}_isic_2024_logs".format("ResNet18", str(criterion).replace('(','').replace(')','')))

    trainer = pl.Trainer(accelerator="gpu",
                         reload_dataloaders_every_n_epochs=1,
                         devices=[0],
                         max_epochs=args["epochs"],
                         logger=tb_logger,
                         precision='16-mixed')



    hparams = {
        'batch_size': args['batch_size'],
        'learning_rate': args["lr"],
        'optimizer': 'Adam',
        'num_epochs': args["epochs"],
        'subset_ratio': args["subset_ratio"],
        'split_ratio': args["split_ratio"],
        'loss': str(criterion)
    }

    #tb_logger.log_hyperparams(hparams,torch.tensor([0],device="cuda:0"))

    trainer.reload_dataloaders_every_epoch = True


    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="_raw_data/train-image.hdf5")
    parser.add_argument("--metadata_path", type=str, default="_raw_data/train-metadata.csv")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--split_ratio", type=int, default=0.8)
    parser.add_argument("--logger_path", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    args = parser.parse_args()
    train(vars(args))
