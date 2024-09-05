import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm import create_model
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from torchmetrics.classification import Accuracy, F1Score, BinaryAccuracy, BinaryF1Score
import numpy as np
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

from utils.plot_functions import plot_confusion_matrix


class EfficientNetV2WithMetadata(pl.LightningModule):
    def __init__(self, num_classes=1,
                 learning_rate=0.001,
                 criterion = F.binary_cross_entropy_with_logits,
                 num_metadata_features=5):
        super(EfficientNetV2WithMetadata, self).__init__()
        #self.save_hyperparameters()

        # Load EfficientNetV2_s with IMAGENET1K_V1 pretrained weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        self.model = efficientnet_v2_s(weights=weights)
        #self.model = efficientnet_v2_s()
        # Freeze the preloaded weights
        for param in self.model.parameters():
            param.requires_grad = False
        #self.model = efficientnet_v2_s()

        # Get the number of input features for the classifier
        num_ftrs = self.model.classifier[1].in_features

        # Replace the classifier with an identity layer to extract features
        self.model.classifier = nn.Identity()

        # Metadata processing layers
        self.metadata_fc = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
        )

        self.batch_norm = nn.BatchNorm1d(num_ftrs + 32)
        #self.layer_norm = nn.LayerNorm(num_ftrs + 32)

        # Combined classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 32, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes)
        )

        # Ensure metadata and classifier layers are trainable
        for param in self.metadata_fc.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

        for param in self.metadata_fc.parameters():
            assert param.requires_grad, "Gradient computation is disabled for metadata_fc"

        # Initialize the new layers
        self._initialize_weights(self.classifier)
        self._initialize_weights(self.metadata_fc)

        self.learning_rate = learning_rate

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

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):


        image, metadata = x
        #metadata_variance = metadata.var(dim=0)
        image_features = self.model(image)
        metadata_features = self.metadata_fc(metadata[:, 0])
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        #combined_features = self.batch_norm(combined_features)
        output = self.classifier(combined_features)
        return output

    def validation_step(self, batch, batch_idx):
        data, labels = batch

        images, metadata = data
        logits = self(data).squeeze(1)
        loss = self.criterion(logits, labels.float())

        preds = (logits > 0.5).float()
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)
        if torch.isnan(loss):
            print("VALIDATION NAN")

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
        #pauc2 = self.custom_metric(self.val_targets, self.val_pred_scores)
        self.log('val_pauc', pauc, on_epoch=True)
        #self.log('val_pauc_2', pauc2, on_epoch=True)
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
        #pauc2 = self.custom_metric(self.targets, self.pred_scores)
        self.log('train_pauc', pauc, on_epoch=True)
        #self.log('train_pauc_2', pauc2, on_epoch=True)
        self.pred_scores.clear()
        self.targets.clear()

    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images).squeeze(1)  # Ensure logits is of shape (batch_size,)
        loss = self.criterion(logits, labels.float())

        preds = torch.sigmoid(logits) > 0.5
        acc = self.test_accuracy(preds, labels)
        f1 = self.test_f1_f1(preds, labels)

        if torch.isnan(loss):
            print("TEST NAN")
        else:
            print("TEST LOSS: {}".format(loss))

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        data, labels = batch

        #images, metadata = data
        logits = self(data).squeeze(1)

        loss = self.criterion(logits, labels.float())

        preds = (logits > 0.5).float()
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        if torch.isnan(loss):
            print("TRAIN NAN")
        else:
            print("TRAIN LOSS: {}".format(loss))

        # Perform backward pass
        #self.manual_backward(loss)

        # Check gradients for the metadata_fc layer
        for name, param in self.metadata_fc.named_parameters():
            if param.grad is not None:
                self.log(f'grad_{name}', param.grad.norm().item())
            else:
                print(f'No gradient for {name}')


        self.pred_scores.extend(logits.detach().cpu().numpy())
        self.targets.extend(labels.detach().cpu().numpy())

        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.log('train_accuracy', acc, on_epoch=True, on_step=True)
        self.log('train_f1', f1, on_epoch=True, on_step=True)
        self.log('lr', self.optimizer.param_groups[0]['lr'], on_epoch=True, on_step=False)


        tn, fp, fn, tp = confusion_matrix(labels.tolist(), preds.tolist(), labels=[0, 1]).ravel()
        self.training_step_cms.append([tn, fp, fn, tp])
        return loss

    def configure_optimizers(self):
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.classifier.parameters(), 'lr': self.learning_rate * 1},
            #{'params': self.metadata_fc.parameters(), 'lr': self.learning_rate * 100},
            {'params': self.metadata_fc.parameters(), 'lr': self.learning_rate * 1},  # Higher LR for metadata_fc
        ], lr=self.learning_rate, weight_decay=1e-4)

        # Example: OneCycleLR scheduler

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # or 'max' for maximizing metrics like accuracy
            factor=0.7,  # Factor by which the learning rate will be reduced
            patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
            verbose=True,  # Print a message to the console each time the learning rate is reduced
            min_lr=1e-6  # Minimum learning rate
        )

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_pauc',  # Metric to monitor
                'interval': 'epoch',  # Interval at which to monitor the metric
                'frequency': 1,  # How often to apply the scheduler
                'name': 'reduce_lr_on_plateau'  # Name for logging purposes
            }
        }

    def on_train_epoch_start(self):
        self.trainer.datamodule.update_train_dataset()

    def on_after_backward(self):
        #print(self.metadata_fc[0].weight.data)
        #print(self.classifier[0].weight.data)
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

    def p_roc_auc_score(self, y_true, y_preds, min_tpr: float = 0.80):
        v_gt = abs(np.asarray(y_true) - 1)
        v_pred = -1.0 * np.asarray(y_preds)
        max_fpr = abs(1 - min_tpr)
        fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)
        return partial_auc

    def custom_metric(self, y_true, y_hat):

        min_tpr = 0.80
        max_fpr = abs(1 - min_tpr)

        v_gt = abs(np.asarray(y_true) - 1)
        v_pred = np.array([1.0 - x for x in y_hat])

        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
        partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

        return partial_auc