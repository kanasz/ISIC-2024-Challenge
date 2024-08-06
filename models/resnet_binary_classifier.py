import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, F1Score


class ResNetBinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=0.001):
        super(ResNetBinaryClassifier, self).__init__()
        self.learning_rate = learning_rate
        # Load a pre-trained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Keep the original fully connected layer
        self.fc1 = self.resnet.fc

        # Define an additional fully connected layer
        self.fc2 = nn.Linear(self.fc1.out_features, num_classes)

        # Replace the original FC layer in ResNet with an Identity layer
        self.resnet.fc = nn.Identity()

        # Initialize accuracy metric
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        # Extract features using the ResNet backbone
        features = self.resnet(x)  # This will be a 2048-dimensional vector

        # Pass through the first fully connected layer (fc1)
        x = self.fc1(features)

        # Apply a non-linearity (e.g., ReLU)
        x = F.relu(x)

        # Pass through the second fully connected layer (fc2)
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_f1', f1)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
