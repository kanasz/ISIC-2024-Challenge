import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, roc_auc_score
from torchmetrics.classification import Accuracy, F1Score, BinaryAccuracy, BinaryF1Score

from utils.plot_functions import plot_confusion_matrix


class ResNetBinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=1, learning_rate=0.001, criterion = F.binary_cross_entropy_with_logits):
        super(ResNetBinaryClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.model = models.resnet18(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.validation_step_cms = []
        self.training_step_cms = []
        self.pred_scores = []
        self.targets = []

        self.val_pred_scores = []
        self.val_targets = []

        self.criterion = criterion

    def setup(self, stage=None):
        return

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)

        loss = self.criterion(logits, labels.float())

        preds = (logits > 0.5).float()
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.pred_scores.extend(logits.detach().cpu().numpy())
        self.targets.extend(labels.detach().cpu().numpy())

        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_accuracy', acc, on_epoch=True, on_step=True)
        self.log('train_f1', f1, on_epoch=True, on_step=True)

        tn, fp, fn, tp = confusion_matrix(labels.tolist(), preds.tolist(), labels=[0, 1]).ravel()
        self.training_step_cms.append([tn, fp, fn, tp])
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0.5).float()
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.val_pred_scores.extend(logits.detach().cpu().numpy())
        self.val_targets.extend(labels.detach().cpu().numpy())

        self.log('val_loss', loss, on_epoch=True, on_step=True)
        self.log('val_accuracy', acc, on_epoch=True, on_step=True)
        self.log('val_f1', f1, on_epoch=True, on_step=True)

        tn, fp, fn, tp = confusion_matrix(labels.tolist(), preds.tolist(), labels=[0, 1]).ravel()
        self.validation_step_cms.append([tn, fp, fn, tp])
        return loss

    def on_validation_epoch_end(self):
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        tp_sum = 0
        for cm in self.validation_step_cms:
            [tn, fp, fn, tp] = cm
            tn_sum = tn_sum + tn
            fp_sum = fp_sum + fp
            fn_sum = fn_sum + fn
            tp_sum = tp_sum + tp

        confusion_matrix = np.array([[tn_sum, fp_sum],
                                     [fn_sum, tp_sum]])

        class_names = ['Benign', 'Malign']

        fig = plot_confusion_matrix(confusion_matrix, class_names)
        epoch_number = self.current_epoch
        self.logger.experiment.add_figure('Validation Set Confusion Matrix', fig, global_step=epoch_number)
        self.validation_step_cms.clear()

        pauc = self.computePAUCV2(self.val_targets, self.val_pred_scores)
        self.log('val_pauc', pauc, on_epoch=True)
        self.val_pred_scores.clear()
        self.val_targets.clear()

    def on_train_epoch_end(self):

        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        tp_sum = 0
        for cm in self.training_step_cms:
            [tn, fp, fn, tp] = cm
            tn_sum = tn_sum + tn
            fp_sum = fp_sum + fp
            fn_sum = fn_sum + fn
            tp_sum = tp_sum + tp

        confusion_matrix = np.array([[tn_sum, fp_sum],
                                     [fn_sum, tp_sum]])

        class_names = ['Benign', 'Malign']

        fig = plot_confusion_matrix(confusion_matrix, class_names)
        epoch_number = self.current_epoch
        self.logger.experiment.add_figure('Training Set Confusion Matrix', fig, global_step=epoch_number)
        self.training_step_cms.clear()

        pauc = self.computePAUCV2(self.targets, self.pred_scores)
        self.log('train_pauc', pauc, on_epoch=True)
        self.pred_scores.clear()
        self.targets.clear()

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images).squeeze(1)  # Ensure logits is of shape (batch_size,)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        preds = torch.sigmoid(logits) > 0.5
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self):
        self.trainer.datamodule.update_train_dataset()

    def on_after_backward(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log('grad_norm', grad_norm)
        for name, param in self.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm()
                self.log(f'gradients/{name}_norm', norm)

                if norm > 10.0:
                    print(f"Warning: Exploding gradient detected in {name}")
                elif norm < 1e-5:
                    print(f"Warning: Vanishing gradient detected in {name}")

    def computePAUCV2(self, targets, predictions, tprThreshold=0.80):
        # Ensure the inputs are numpy arrays for processing
        targets = np.array(targets)
        predictions = np.array(predictions)

        v_gt = np.abs(targets - 1)  # Assuming 'targets' are 0s and 1s
        v_pred = 1.0 - predictions  # Inverting predictions if necessary

        max_fpr = np.abs(1 - tprThreshold)
        try:
            partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
        except:
            print('ERROR')
            return 0

        # Adjust scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
        partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
        return partial_auc
