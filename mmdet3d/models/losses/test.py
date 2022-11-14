from mmdet.models.losses import FocalLoss, sigmoid_focal_loss

import numpy as np
import torch

if __name__ == '__main__':
    loss_fn = FocalLoss(use_sigmoid=True)
    net_out_put = torch.randn(size=(8, 40, 200))
    label = np.random.randint(1, size=(8, 40, 200))
    label = torch.from_numpy(label)
    ce = loss_fn(net_out_put.reshape(-1, 1), label.reshape(-1))
    # ce = sigmoid_focal_loss(net_out_put.reshape(-1, 1), label.reshape(-1, 1))
    print(ce)