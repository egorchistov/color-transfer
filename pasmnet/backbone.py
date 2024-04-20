import torch


class ResB(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)
