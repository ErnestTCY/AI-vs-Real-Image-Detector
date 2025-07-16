# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def build_model(config, device=torch.device("cpu"), return_optimizer=True):
    # Load backbone
    model = models.efficientnet_b0(weights=config['weights']).to(device)

    # Freeze if only output head should train
    if config['finetuned_layers'] == 'output':
        for p in model.parameters():
            p.requires_grad = False

    # Replace classifier with [Dropout, Linear→256, ReLU, Dropout, Linear→1, Sigmoid]
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1280, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=256, out_features=1),
        nn.Sigmoid()
    ).to(device)

    # Build optimizer over all trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=config['lr']) \
          if config['optimizer'].lower()=='adam' \
          else optim.SGD(trainable, lr=config['lr'], momentum=config['momentum'])

    if return_optimizer:
        return model, opt
    return model
