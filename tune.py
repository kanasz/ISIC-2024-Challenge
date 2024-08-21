import argparse
import albumentations as A
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data.medical_data_module import MedicalDataModule
from loss_functions.ldam_loss import LDAMLoss
from models.resnet_binary_classifier import ResNetBinaryClassifier

torch.set_float32_matmul_precision('medium')


def objective(trial, batch_size, image_path, metadata_path, split_ratio, subset_ratio, epochs, logger_path):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ldam_loss_s = trial.suggest_int('s', 1, 60)
    transforms = {
        "train": A.Compose([
            A.Resize(height=224, width=224),
            # A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()
        ]),
        "validation": A.Compose([
            A.Resize(height=224, width=224),
            # A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ]),
    }

    dm = MedicalDataModule(
        image_path=image_path,
        metadata_path=metadata_path,
        batch_size=batch_size,
        split_ratio=split_ratio,
        transforms=transforms,
        subset_ratio=subset_ratio,
        training_minority_oversampling_ceoff=2
    )
    dm.setup()
    pos_weight = dm.pos_weight

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    w1 = dm.cls_num_list[1] / (dm.cls_num_list[0] + dm.cls_num_list[1])
    w2 = dm.cls_num_list[0] / (dm.cls_num_list[0] + dm.cls_num_list[1])
    #criterion = LDAMLoss(dm.cls_num_list, weight=torch.tensor([w1, w2], device="cuda"), s=ldam_loss_s)

    tb_logger = TensorBoardLogger(logger_path,
                                  name="ResNet_{}_isic_2024_logs".format(str(criterion).replace('(', '').replace(')', '')))

    trainer = pl.Trainer(accelerator="gpu",
                         reload_dataloaders_every_n_epochs=1,
                         devices=[0],
                         max_epochs=epochs,
                         logger=tb_logger,
                         precision='16-mixed')



    trainer.reload_dataloaders_every_epoch = True

    model = ResNetBinaryClassifier(learning_rate=learning_rate, criterion=criterion)

    trainer.fit(model, datamodule=dm)

    metric = trainer.callback_metrics["val_pauc"].item()
    hparams = {
        #'batch_size': batch_size,
        'learning_rate': learning_rate,

        's':ldam_loss_s,
        #'optimizer': 'Adam',
        #'num_epochs': epochs,
        #'subset_ratio': subset_ratio,
        #'split_ratio': split_ratio,
        'loss': str(criterion),
        'hp_metric': metric
    }

    tb_logger.log_hyperparams(hparams, metric)

    return metric


def tune(args):
    batch_size = args['batch_size']
    image_path = args['image_path']
    metadata_path = args['metadata_path']
    split_ratio = args['split_ratio']
    subset_ratio = args['subset_ratio']
    epochs = args['epochs']
    logger_path = args['logger_path']
    # Run the Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, batch_size, image_path, metadata_path, split_ratio, subset_ratio, epochs, logger_path), n_trials=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="_raw_data/train-image.hdf5")
    parser.add_argument("--metadata_path", type=str, default="_raw_data/train-metadata.csv")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--split_ratio", type=int, default=0.8)
    parser.add_argument("--logger_path", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--subset_ratio", type=float, default=0.01)
    args = parser.parse_args()
    tune(vars(args))
