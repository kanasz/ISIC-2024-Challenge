import argparse
import albumentations as A
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import random
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import EfficientNet_V2_S_Weights

from data.medical_data_module import MedicalDataModule
from loss_functions.ldam_loss import LDAMLoss
from models.efficientnet_binary_classifier import EfficientNetBinaryClassifier
from models.efficientnetv2_binary_classifier import EfficientNetV2WithMetadata
from models.resnet_binary_classifier import ResNetBinaryClassifier
from loss_functions.focal_loss import FocalLoss

torch.set_float32_matmul_precision('medium')
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def objective(trial, batch_size, image_path, metadata_path, split_ratio, subset_ratio, epochs, logger_path, learning_rate):

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-5)

    training_minority_oversampling_ceoff = trial.suggest_int('training_minority_oversampling_ceoff', 20, 30)
    gamma = trial.suggest_int('gamma', 1, 5)
    alpha = trial.suggest_loguniform('alpha', 0.001, 1)

    print("LR: {}, ".format(learning_rate))
    #alpha = None
    #gamma = None
    #ldam_loss_s = trial.suggest_int('s', 1, 10)
    ldam_loss_s = None
    print( "LR: {}, Oversamp. Coeff.: {}, gamma: {}, alpha: {}".format(learning_rate, training_minority_oversampling_ceoff, gamma, alpha))
    print("LR: {}, Oversamp. Coeff.: {}, s: {}".format(learning_rate, training_minority_oversampling_ceoff, ldam_loss_s))
    transforms = {
        "train": A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.15, rotate_limit=90, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.12, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.0, p=0.7),

            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

            #A.ElasticTransform(alpha=0.2, sigma=6.0, alpha_affine=20.0, p=0.5),
            A.GridDistortion(num_steps=2, distort_limit=0.2, p=0.5),
            A.Resize(128, 128),
            A.Normalize(mean=EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms().mean,
                                 std=EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms().std),
            #A.Resize(224, 224),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        "validation": A.Compose([
            A.Resize(128, 128),
            A.Normalize(mean=EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms().mean,
                        std=EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms().std),
            #A.Resize(height=224, width=224),
            #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        training_minority_oversampling_ceoff=training_minority_oversampling_ceoff,
        lesion_confidence_threshold=50
    )
    dm.setup()
    pos_weight = dm.pos_weight
    weights = dm.weights

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = FocalLoss(gamma=gamma, alpha=alpha, weight=pos_weight)
    #criterion=nn.BCEWithLogitsLoss()
    w1 = dm.cls_num_list[1] / (dm.cls_num_list[0] + dm.cls_num_list[1])
    w2 = dm.cls_num_list[0] / (dm.cls_num_list[0] + dm.cls_num_list[1])
    #criterion = LDAMLoss(dm.cls_num_list, weight=torch.tensor([w1, w2], device="cuda"), s=ldam_loss_s)

    model = EfficientNetV2WithMetadata(learning_rate=learning_rate, criterion=criterion, num_metadata_features=dm.processed_features_shape[1])
    #model = EfficientNetBinaryClassifier(learning_rate=learning_rate, criterion=criterion, num_metadata_features=dm.processed_features_shape[1])
    model_name = model.__class__.__name__
    tb_logger = TensorBoardLogger(logger_path,
                                  name="{}_{}_logs".format(model_name,
                                                           criterion.__class__.__name__.replace('(', '').replace(')',
                                                                                                                 '')))

    checkpoint_path = '{}_{}_'.format(model_name, criterion.__class__.__name__) + '{epoch:02d}-{val_pauc:.2f}'
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',  # Directory to save the checkpoints
        filename=checkpoint_path,  # Filename format (can be customized)
        save_top_k=1,  # Save only the best checkpoint based on the monitored metric
        monitor='val_pauc',  # Metric to monitor
        mode='max',  # Mode can be 'min' for minimizing or 'max' for maximizing the monitored metric
        save_last=True  # Save the last checkpoint at the end of training
    )

    trainer = pl.Trainer(accelerator="gpu",
                         reload_dataloaders_every_n_epochs=1,
                         devices=[0],
                         max_epochs=epochs,
                         logger=tb_logger,
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=0,
                         precision='16-mixed')

    trainer.reload_dataloaders_every_epoch = True

    trainer.fit(model, datamodule=dm)

    metric = trainer.callback_metrics["val_pauc"].item()
    hparams = {
        'loss': str(criterion),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'oversampling_coef': training_minority_oversampling_ceoff,
        'alpha': alpha,
        'gamma': gamma,
        's': ldam_loss_s,
        'subset_ratio':subset_ratio,
        'hp_metric': metric,
        'training_minority_oversampling_ceoff':training_minority_oversampling_ceoff,
        'epochs':epochs
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
    learning_rate = args['lr']
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, batch_size, image_path, metadata_path, split_ratio, subset_ratio, epochs,
                                logger_path, learning_rate), n_trials=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="_raw_data/train-image.hdf5")
    parser.add_argument("--metadata_path", type=str, default="_raw_data/train-metadata.csv")
    #parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--split_ratio", type=int, default=0.8)
    parser.add_argument("--logger_path", type=str, default="logs")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=30)
    #parser.add_argument("--subset_ratio", type=float, default=0.05)
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    args = parser.parse_args()
    tune(vars(args))
