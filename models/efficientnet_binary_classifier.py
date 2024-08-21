import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, roc_auc_score
from torchmetrics.classification import Accuracy, F1Score, BinaryAccuracy, BinaryF1Score
from efficientnet_pytorch import EfficientNet
from utils.plot_functions import plot_confusion_matrix


class EfficientNetBinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=1, learning_rate=0.001, criterion = F.binary_cross_entropy_with_logits):
        super(EfficientNetBinaryClassifier, self).__init__()
        self.learning_rate = learning_rate
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

        self.model._fc = torch.nn.Linear(self.model._fc.in_features, 1)
        self.lr = learning_rate
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

        #self.criterion = F.binary_cross_entropy_with_logits
        self.criterion = criterion
        #self.pos_weight = pos_weight#torch.tensor([0], device="cuda")

    def setup(self, stage=None):
        return

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)  # Ensure logits is of shape (batch_size,)


        #loss = self.criterion(logits, labels.float(), pos_weight=self.pos_weight)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0.5).float()
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.pred_scores.extend(logits.detach().cpu().numpy())
        self.targets.extend(labels.detach().cpu().numpy())

        #self.log('train_pauc', pauc2, on_epoch=True, on_step=True)

        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_accuracy', acc, on_epoch=True, on_step=True)
        self.log('train_f1', f1, on_epoch=True, on_step=True)

        tn, fp, fn, tp = confusion_matrix(labels.tolist(), preds.tolist(), labels=[0, 1]).ravel()
        self.training_step_cms.append([tn, fp, fn, tp])
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(1)  # Ensure logits is of shape (batch_size,)
        #loss = self.criterion(logits, labels.float(), pos_weight=self.pos_weight)
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
        #self.pos_weight = self.trainer.datamodule.pos_weight
        #    pos_weight = torch.tensor([num_negatives / num_positives])

    def on_after_backward(self):
        # Log gradient norms for all parameters
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



    '''
    def score_pauc(self, solution, submission, min_tpr: float = 0.80) -> float:
        v_gt = torch.abs(solution-1)
        v_pred = -1.0*submission
        max_fpr = torch.abs(torch.tensor([1 - min_tpr], device='cuda:0'))
        partial_auc_scaled = roc_auc_score(v_gt.cpu().numpy(), v_pred.cpu().numpy(), max_fpr=max_fpr.cpu().numpy()[0])
        # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
        # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
        partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

        return (partial_auc)
    '''

    def compute_roc(self, y_true, y_scores):
        # Sort scores and corresponding true labels
        sorted_indices = torch.argsort(y_scores, descending=True)
        y_true_sorted = y_true[sorted_indices]
        y_scores_sorted = y_scores[sorted_indices]

        # Compute cumulative sum of true positives and false positives
        tps = torch.cumsum(y_true_sorted, dim=0)
        fps = torch.cumsum(1 - y_true_sorted, dim=0)

        # Compute TPR and FPR
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Add (0, 0) and (1, 1) to the curve
        tpr = torch.cat([torch.tensor([0.0], device='cuda:0'), tpr, torch.tensor([1.0], device='cuda:0')])
        fpr = torch.cat([torch.tensor([0.0], device='cuda:0'), fpr, torch.tensor([1.0], device='cuda:0')])

        return fpr, tpr

    def compute_pAUC(self, y_true, y_scores, tpr_min):
        # Compute ROC curve
        fpr, tpr = self.compute_roc(y_true, y_scores)

        # Identify the FPR range where TPR is at least tpr_min
        mask = tpr >= tpr_min
        fpr_selected = fpr[mask]
        tpr_selected = tpr[mask]

        # Ensure the integration limits start from TPR of tpr_min
        if tpr_selected[0] > tpr_min:
            prev_idx = (tpr < tpr_min).nonzero(as_tuple=True)[0][-1]
            fpr_start = torch.lerp(fpr[prev_idx], fpr[prev_idx + 1],
                                   (tpr_min - tpr[prev_idx]) / (tpr[prev_idx + 1] - tpr[prev_idx]))
            fpr_selected = torch.cat([torch.tensor([fpr_start], device='cuda:0'), fpr_selected])
            tpr_selected = torch.cat([torch.tensor([tpr_min], device='cuda:0'), tpr_selected])

        # Compute the partial AUC using the trapezoidal rule
        pAUC = torch.trapz(tpr_selected, fpr_selected)

        # Normalize the pAUC
        fpr_range = fpr_selected[-1] - fpr_selected[0]
        pAUC /= fpr_range

        return pAUC

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
