import torch
import torch.nn.functional as F

from tools import set_device

# Set the device for computation (CPU or GPU)
device = set_device()

# Loss M
# Initialize Mean Squared Error Loss function
loss_M = torch.nn.MSELoss()

# Loss mc
# Constants for filters
d = 1 / (2**0.5)
# Define filters for computing gradients in X and Y directions
filter_dX = torch.tensor([[-d, 0, d], [-1, 0, 1], [-d, 0, d]], dtype=torch.float32).to(
    device
)
filter_dY = torch.tensor([[-d, -1, -d], [0, 0, 0], [d, 1, d]], dtype=torch.float32).to(
    device
)
# Reshape filters to 4D for use with conv2d
filter_dX = filter_dX.view(1, 1, 3, 3)
filter_dY = filter_dY.view(1, 1, 3, 3)


def loss_mc(preds_Nsigma_C):
    """
    Compute the loss_mc which enforces mechanical equilibrium by minimizing the divergence of the predicted stresses.

    Args:
    preds_Nsigma_C (torch.Tensor): Predicted stresses with shape (batch_size, channels, height, width).
        - 0: sigma_xx
        - 1: sigma_xy
        - 2: sigma_yy

    Returns:
    torch.Tensor: Computed loss_mc value.
    """
    # 0 - sigma_xx | 1 - sigma_xy | 2 - sigma_yy
    # Compute derivatives of stress components
    dsigmaxx_dx = F.conv2d(preds_Nsigma_C[:, 0:1, :, :], filter_dX, padding=1)
    dsigmaxy_dy = F.conv2d(preds_Nsigma_C[:, 1:2, :, :], filter_dY, padding=1)

    dsigmayy_dy = F.conv2d(preds_Nsigma_C[:, 2:3, :, :], filter_dY, padding=1)
    dsigmaxy_dx = F.conv2d(preds_Nsigma_C[:, 1:2, :, :], filter_dX, padding=1)
    # Crop the borders to remove padding artifacts
    crop = 2
    loss_mc1 = (dsigmaxx_dx + dsigmaxy_dy)[:, :, crop:-crop, crop:-crop]
    loss_mc2 = (dsigmaxy_dx + dsigmayy_dy)[:, :, crop:-crop, crop:-crop]
    # Compute the mean squared error of the sum of derivatives
    return torch.mean(torch.square(loss_mc1)) + torch.mean(torch.square(loss_mc2))


# Loss seg
def loss_seg(
    epsilon_raw_C,
    preds_Nsigma_C,
    preds_Nseg_C,
    E_tensor,
    poisson_tensor,
):
    """
    Compute the segmentation loss which aligns the predicted stresses and segmentation with the stress-strain relationship.

    Args:
    epsilon_raw_C (torch.Tensor): Input strain tensor with shape (batch_size, channels, height, width).
    preds_Nsigma_C (torch.Tensor): Predicted stresses with shape (batch_size, channels, height, width).
    preds_Nseg_C (torch.Tensor): Predicted segmentation mask with shape (batch_size, channels, height, width).
    E_tensor (torch.Tensor): Young's modulus tensor.
    poisson_tensor (torch.Tensor): Poisson's ratio tensor.

    Returns:
    torch.Tensor: Computed segmentation loss value.
    """
    epsilon_xx = epsilon_raw_C[:, 0:1, :, :]
    epsilon_xy = epsilon_raw_C[:, 1:2, :, :]
    epsilon_yy = epsilon_raw_C[:, 2:3, :, :]

    sigma_xx_from_Nsigma = preds_Nsigma_C[:, 0:1, :, :]
    sigma_xy_from_Nsigma = preds_Nsigma_C[:, 1:2, :, :]
    sigma_yy_from_Nsigma = preds_Nsigma_C[:, 2:3, :, :]

    # Compute predicted stresses based on segmentation
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
    # Compute the mean squared error of the differences
    loss_xx = torch.square(sigma_xx_from_Nseg - sigma_xx_from_Nsigma)
    loss_xy = torch.square(sigma_xy_from_Nseg - sigma_xy_from_Nsigma)
    loss_yy = torch.square(sigma_yy_from_Nseg - sigma_yy_from_Nsigma)
    # Weight the loss by the log of the inverse of the predicted segmentation mask
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
    """
    Compute the loss_phi which aligns the predicted stresses with the stress-strain relationship calculated by the logistic-transformed segmentation mask.

    Args:
    epsilon_raw (torch.Tensor): Input strain tensor with shape (batch_size, channels, height, width).
    preds_Nsigma_C (torch.Tensor): Predicted stresses with shape (batch_size, channels, height, width).
    preds_Nseg_C (torch.Tensor): Predicted segmentation mask with shape (batch_size, channels, height, width).
    E_tensor (torch.Tensor): Young's modulus tensor.
    poisson_tensor (torch.Tensor): Poisson's ratio tensor.

    Returns:
    torch.Tensor: Computed phi loss value.
    """
    epsilon_xx = epsilon_raw[:, 0:1, :, :]
    epsilon_xy = epsilon_raw[:, 1:2, :, :]
    epsilon_yy = epsilon_raw[:, 2:3, :, :]

    sigma_xx_from_Nsigma = preds_Nsigma_C[:, 0, :, :]
    sigma_xy_from_Nsigma = preds_Nsigma_C[:, 1, :, :]
    sigma_yy_from_Nsigma = preds_Nsigma_C[:, 2, :, :]

    # Compute predicted stresses based on segmentation
    sigma_xx_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_xx
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
    )
    sigma_xy_from_Nseg = (
        2 * (E_tensor / (2 * (1 + poisson_tensor))) * epsilon_xy
    ) 
    sigma_yy_from_Nseg = (
        (E_tensor / (1 + poisson_tensor))
        * (
            epsilon_yy
            + (poisson_tensor / (1 - 2 * poisson_tensor)) * (epsilon_xx + epsilon_yy)
        )
    )
    # Apply logistic transformation to segmentation mask
    preds_Nseg_logistic = 1 / (1 + torch.exp(-10 * (preds_Nseg_C - 0.5)))

    sigma_xx_from_Nseg = torch.sum(sigma_xx_from_Nseg * preds_Nseg_logistic, axis=1)
    sigma_xy_from_Nseg = torch.sum(sigma_xy_from_Nseg * preds_Nseg_logistic, axis=1)
    sigma_yy_from_Nseg = torch.sum(sigma_yy_from_Nseg * preds_Nseg_logistic, axis=1)
    
    # Compute the mean squared error of the differences
    loss_xx = torch.square(sigma_xx_from_Nseg - sigma_xx_from_Nsigma)
    loss_xy = torch.square(sigma_xy_from_Nseg - sigma_xy_from_Nsigma)
    loss_yy = torch.square(sigma_yy_from_Nseg - sigma_yy_from_Nsigma)

    return torch.mean(loss_xx + loss_xy + loss_yy)
