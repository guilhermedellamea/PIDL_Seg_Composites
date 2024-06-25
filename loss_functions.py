import torch
import torch.nn.functional as F

from tools import set_device

device = set_device()

# Loss M
loss_M = torch.nn.MSELoss()

# Loss mc
d = 1 / (2**0.5)

filter_dX = torch.tensor([[-d, 0, d], [-1, 0, 1], [-d, 0, d]], dtype=torch.float32).to(
    device
)
filter_dY = torch.tensor([[-d, -1, -d], [0, 0, 0], [d, 1, d]], dtype=torch.float32).to(
    device
)

filter_dX = filter_dX.view(1, 1, 3, 3)
filter_dY = filter_dY.view(1, 1, 3, 3)


def loss_mc(preds_Nsigma_C):

    # 0 - sigma_xx | 1 - sigma_xy | 2 - sigma_yy

    dsigmaxx_dx = F.conv2d(preds_Nsigma_C[:, 0:1, :, :], filter_dX, padding=1)
    dsigmaxy_dy = F.conv2d(preds_Nsigma_C[:, 1:2, :, :], filter_dY, padding=1)

    dsigmayy_dy = F.conv2d(preds_Nsigma_C[:, 2:3, :, :], filter_dY, padding=1)
    dsigmaxy_dx = F.conv2d(preds_Nsigma_C[:, 1:2, :, :], filter_dX, padding=1)

    crop = 2
    loss_mc1 = (dsigmaxx_dx + dsigmaxy_dy)[:, :, crop:-crop, crop:-crop]
    loss_mc2 = (dsigmaxy_dx + dsigmayy_dy)[:, :, crop:-crop, crop:-crop]

    return torch.mean(torch.square(loss_mc1)) + torch.mean(torch.square(loss_mc2))


# Loss seg
def loss_seg(
    epsilon_raw_C,
    preds_Nsigma_C,
    preds_Nseg_C,
    E_tensor,
    poisson_tensor,
):
    epsilon_xx = epsilon_raw_C[:, 0:1, :, :]
    epsilon_xy = epsilon_raw_C[:, 1:2, :, :]
    epsilon_yy = epsilon_raw_C[:, 2:3, :, :]

    sigma_xx_from_Nsigma = preds_Nsigma_C[:, 0:1, :, :]
    sigma_xy_from_Nsigma = preds_Nsigma_C[:, 1:2, :, :]
    sigma_yy_from_Nsigma = preds_Nsigma_C[:, 2:3, :, :]

    sigma_xx_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_xx
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
        * preds_Nseg_C
    )
    sigma_xy_from_Nseg = (
        2 * (E_tensor / (2 * (1 + poisson_tensor))) * epsilon_xy
    ) * preds_Nseg_C
    sigma_yy_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_yy
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
        * preds_Nseg_C
    )

    loss_xx = torch.square(sigma_xx_from_Nseg - sigma_xx_from_Nsigma)
    loss_xy = torch.square(sigma_xy_from_Nseg - sigma_xy_from_Nsigma)
    loss_yy = torch.square(sigma_yy_from_Nseg - sigma_yy_from_Nsigma)

    return torch.mean(
        torch.sum(
            (loss_xx + loss_xy + loss_yy) * -torch.log(1 - preds_Nseg_C + 1e-7), axis=1
        )
    )


# Loss phi
def loss_phi(
    epsilon_raw,
    preds_Nsigma_C,
    preds_Nseg_C,
    E_tensor,
    poisson_tensor,
):
    epsilon_xx = epsilon_raw[:, 0:1, :, :]
    epsilon_xy = epsilon_raw[:, 1:2, :, :]
    epsilon_yy = epsilon_raw[:, 2:3, :, :]

    sigma_xx_from_Nsigma = preds_Nsigma_C[:, 0, :, :]
    sigma_xy_from_Nsigma = preds_Nsigma_C[:, 1, :, :]
    sigma_yy_from_Nsigma = preds_Nsigma_C[:, 2, :, :]

    sigma_xx_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_xx
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
        * preds_Nseg_C
    )
    sigma_xy_from_Nseg = (
        2 * (E_tensor / (2 * (1 + poisson_tensor))) * epsilon_xy
    ) * preds_Nseg_C
    sigma_yy_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_yy
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
        * preds_Nseg_C
    )

    preds_Nseg_logistic = 1 / (1 + torch.exp(-10 * (preds_Nseg_C - 0.5)))

    sigma_xx_from_Nseg = torch.sum(sigma_xx_from_Nseg * preds_Nseg_logistic, axis=1)
    sigma_xy_from_Nseg = torch.sum(sigma_xy_from_Nseg * preds_Nseg_logistic, axis=1)
    sigma_yy_from_Nseg = torch.sum(sigma_yy_from_Nseg * preds_Nseg_logistic, axis=1)

    loss_xx = torch.square(sigma_xx_from_Nseg - sigma_xx_from_Nsigma)
    loss_xy = torch.square(sigma_xy_from_Nseg - sigma_xy_from_Nsigma)
    loss_yy = torch.square(sigma_yy_from_Nseg - sigma_yy_from_Nsigma)

    return torch.mean(loss_xx + loss_xy + loss_yy)
