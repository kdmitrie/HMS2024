import torch
import timm


class HMSSimpleNet(torch.nn.Module):
    def __init__(self, device: torch.device, model_name: str = 'resnet18'):
        super(HMSSimpleNet, self).__init__()
        self.bb1 = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='', in_chans=4)
        self.bb2 = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='', in_chans=1)
        self.flatten = torch.nn.Flatten(start_dim=-3)

        self.head1 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(2048),
            torch.nn.Linear(in_features=2048, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32),
            torch.nn.ReLU(),
        )

        self.head2 = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(4096),
            torch.nn.Linear(in_features=4096, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32),
            torch.nn.ReLU(),
        )

        self.proc1 = torch.nn.Sequential(
            self.bb1,
            self.flatten,
            self.head1
        )

        self.proc2 = torch.nn.Sequential(
            self.bb2,
            self.flatten,
            self.head2
        )

        self.classificator = torch.nn.Sequential(
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=6),
            # torch.nn.Softmax(dim=-1)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        sg, eeg = x
        sg = sg.to(self.device)
        eeg = eeg.to(self.device)

        sg = self.proc1(sg)
        eeg = self.proc2(eeg)
        combined = torch.cat((sg, eeg), axis=1)
        return self.classificator(combined)
