import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl


class ResNetBinaryClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=0.001):
        super(ResNetBinaryClassifier, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Load a pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)

        # Modify the final fully connected layer to output the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
