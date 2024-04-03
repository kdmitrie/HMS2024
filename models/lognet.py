import torch
import timm
import numpy as np


class LogNetLR(torch.nn.Module):
    epsilon = float = 10 ** -15

    def __init__(self, device: torch.device, model_name: str = 'tf_efficientnet_b0'):
        super(LogNetLR, self).__init__()
        self._model = timm.create_model(model_name, pretrained=True, num_classes=6, in_chans=1)
        self.device = device
        self.to(device)

    def forward(self, x):
        sg, eeg = x

        data = np.concatenate((sg, eeg), axis=2)
        data = np.moveaxis(data, (2, 3), (3, 2))
        data = torch.from_numpy(data).to(self.device)

        logits = self._model(data)

        probs = torch.nn.Softmax(dim=-1)(logits)
        probs = torch.clip(probs, min=self.epsilon, max=1 - self.epsilon)
        log_probs = torch.log(probs)
        return log_probs

    def get_embedding(self, x):
        sg, eeg = x

        data = np.concatenate((sg, eeg), axis=2)
        data = np.moveaxis(data, (2, 3), (3, 2))
        data = torch.from_numpy(data).to(self.device)

        return self._model[:-1](data)

    def to_sequential(self):
        fc = self._model.classifier
        self._model.classifier = torch.nn.Identity()
        self._model = torch.nn.Sequential(self._model, fc)


class LogNetLR1(LogNetLR):
    def __init__(self, device: torch.device, model_name: str = 'tf_efficientnet_b0'):
        super().__init__(device, model_name)
        model = self._model
        self._model = torch.nn.Sequential(
            *list(model.children())[:-2],
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(model.num_features, 6)
        )

    def to_sequential(self):
        pass


class LogNetLR2(LogNetLR):
    def __init__(self, device: torch.device, model_name: str = 'tf_efficientnet_b1'):
        super().__init__(device, model_name)
