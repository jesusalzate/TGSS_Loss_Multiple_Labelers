import torch
import torch.nn as nn
import torch.nn.functional as F

class TcgeConfig_pytorch_3D:
    """
    TCGE configuration parameters.
    """
    def __init__(self, num_annotators=5, num_classes=2, gamma=0.1):
        self.num_annotators = num_annotators
        self.num_classes = num_classes
        self.gamma = gamma

class TcgeSs_pytorch_3D(nn.Module):
    """
    Truncated generalized cross entropy for semantic segmentation loss.
    """
    def __init__(self, config, q=0.1, smooth=1e-5):
        super(TcgeSs_pytorch_3D, self).__init__()
        self.q = q
        self.num_annotators = config.num_annotators
        self.num_classes = config.num_classes
        self.smooth = smooth
        self.gamma = config.gamma

    def forward(self, y_true, y_pred, device):
        #y_true = y_true.to(torch.float32)
        #y_pred = y_pred.to(torch.float32)

        y_true = y_true.permute(0,2,3,4,1)
        y_pred = y_pred.permute(0,2,3,4,1)


        y_pred = y_pred[..., :self.num_classes + self.num_annotators]

        

        y_true = y_true.view(y_true.shape[:-1] + (self.num_classes, self.num_annotators))

        lambda_r = y_pred[..., self.num_classes:]
        y_pred_ = y_pred[..., :self.num_classes]
        n_samples,depth, width, height, _ = y_pred_.shape

        y_pred_ = y_pred_.unsqueeze(-1)
        y_pred_ = y_pred_.repeat(1, 1, 1, 1, 1, self.num_annotators)

        epsilon = 1e-8
        y_pred_ = torch.clamp(y_pred_, epsilon, 1.0 - epsilon)

        term_r = torch.mean(
            torch.mul(
                y_true,
                (torch.ones((n_samples,depth, width, height, self.num_classes, self.num_annotators)).to(device) - torch.pow(y_pred_, self.q))
                / (self.q + epsilon + self.smooth)
            ),
            dim=-2
        )

        term_c = torch.mul(
            torch.ones((n_samples,depth, width, height, self.num_annotators)).to(device) - lambda_r,
            (
                torch.ones((n_samples,depth, width, height, self.num_annotators)).to(device)
                - torch.pow((1 / self.num_classes + self.smooth) * torch.ones((n_samples,depth, width, height, self.num_annotators)).to(device), self.q)
            )
            / (self.q + epsilon + self.smooth)
        )

        loss = torch.mean(torch.mul(lambda_r, term_r) + term_c)
        loss = torch.where(torch.isnan(loss), torch.tensor(1e-8, dtype=loss.dtype, device=loss.device), loss)

        return loss

