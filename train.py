import pickle
from datetime import datetime
from pathlib import Path

import torch

from data.datasets_pytorch import get_dataloaders
from loss_functions import loss_M, loss_mc, loss_phi, loss_seg
from networks.UNetPytorch import UNet
from tools import (
    LROnStagnationPlateau,
    MetricsChainedScheduler,
    expand_properties_tensor,
    myprint,
    set_device,
)


# Load the train/validation/test data from a pickle file
with open(f"./data/dataset.pkl", "rb") as file:
    dict_data = pickle.load(file)

# Set the device to GPU if available, otherwise CPU
device = set_device()


# Get PyTorch DataLoaders for training and validation data
dataloaders = get_dataloaders(
    dict_data,
    device=device,
    batch_size_train=32,
    batch_size_validation=2,
    num_workers=0,
)

# Initialize the Nsigma model with specified parameters
Nsigma = UNet(
    n_channels=4,
    n_output_channels=3,
    initial_channels=32,
    ndepth=5,
    bilinear=False,
    activation="elu",
    dropout_rate=None,
).to(device)

# Initialize the Nseg model with specified parameters
Nseg = UNet(
    n_channels=1,
    n_output_channels=3,
    initial_channels=16,
    ndepth=5,
    bilinear=False,
    activation="relu",
    dropout_rate=0.1,
    final_activation="softmax",
).to(device)


# Define optimizers and learning rate schedulers for Nsigma
initial_Nsigma_lr = 1e-3
optimizer_Nsigma = torch.optim.Adam(
    Nsigma.parameters(),
    initial_Nsigma_lr,
    weight_decay=1e-5,
)
lrs_Nsigma1 = LROnStagnationPlateau(
    optimizer_Nsigma,
    patience=5,
    threshold_mode="abs",
    threshold=1e-5,
    min_lr=1e-4,
)
patience_lrs_Nsigma2 = 7
lrs_Nsigma2 = LROnStagnationPlateau(
    optimizer_Nsigma,
    patience=patience_lrs_Nsigma2 + 1,
    threshold_mode="abs",
    threshold=1e-5,
    min_lr=1e-4,
)
scheduler_Nsigma = MetricsChainedScheduler([lrs_Nsigma1, lrs_Nsigma2])

# Define optimizers and learning rate schedulers for Nseg
initial_Nseg_lr = 1e-3
optimizer_Nseg = torch.optim.Adam(
    Nseg.parameters(),
    lr=initial_Nseg_lr,
    weight_decay=1e-5,
)
lrs_Nseg1 = LROnStagnationPlateau(
    optimizer_Nseg,
    patience=5,
    threshold_mode="abs",
    threshold=5e-5,
    min_lr=1e-4,
)
patience_lrs_Nseg2 = patience_lrs_Nsigma2
lrs_Nseg2 = LROnStagnationPlateau(
    optimizer_Nseg,
    patience=patience_lrs_Nseg2 + 1,
    threshold_mode="abs",
    threshold=5e-5,
    min_lr=1e-4,
)

scheduler_Nseg = MetricsChainedScheduler([lrs_Nseg1, lrs_Nseg2])


# Initialize training variables
save = True
segmentation_phase = False
loss_phi_phase = False

best_vloss = best_seg_vloss = 1e4
scheduler_Nsigma.step(best_vloss)
scheduler_Nseg.step(best_seg_vloss)
EPOCHS = 10000
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Path("./models/").mkdir(exist_ok=True)
alpha_loss_mc = 0.1
alpha_loss_phi = 1
E_tensor = expand_properties_tensor(torch.tensor([10.2170,  95.7846, 191.5692]))
poisson_tensor = expand_properties_tensor(torch.tensor([0.3500, 0.2000, 0.2000]))


# Train the Nsigma model for one epoch
def train_Nsigma_one_epoch():
    Nsigma.train()
    running_loss = 0.0
    for iteration, (
        (input_sigma_M, labels_sigma_M),
        (input_sigma_C, epsilon_raw),
    ) in enumerate(
        zip(
            dataloaders["train_sigma_M"],
            dataloaders["train_sigma_C"],
        )
    ):
        optimizer_Nsigma.zero_grad()
        if not loss_phi_phase:
            preds_Nsigma_M = Nsigma(input_sigma_M.to(device))
            loss = loss_M(preds_Nsigma_M, labels_sigma_M.to(device))

            preds_Nsigma_C = Nsigma(input_sigma_C.to(device))
            loss += alpha_loss_mc * loss_mc(preds_Nsigma_C)
        else:
            with torch.no_grad():
                input_seg_C = input_sigma_C[:, :1]
                preds_Nseg_C = Nseg(input_seg_C.to(device))
            preds_Nsigma_C = Nsigma(input_sigma_C.to(device))
            loss = loss_phi(
                epsilon_raw.to(device),
                preds_Nsigma_C,
                preds_Nseg_C,
                E_tensor,
                poisson_tensor,
            )

        loss.backward()
        optimizer_Nsigma.step()
        running_loss += loss.item()

    return running_loss / (iteration + 1)

# Validate the models for one epoch
def validation_one_epoch():
    Nseg.eval()
    running_vloss = running_seg_vloss = 0.0
    for iteration, (
        (input_sigma_M, labels_sigma_M),
        (input_sigma_C, epsilon_raw),
    ) in enumerate(
        zip(
            dataloaders["validation_sigma_M"],
            dataloaders["validation_sigma_C"],
        )
    ):

        preds_Nsigma_C = Nsigma(input_sigma_C.to(device))

        if segmentation_phase:
            input_seg_C = input_sigma_C[:, :1]
            preds_Nseg_C = Nseg(input_seg_C.to(device))
            loss_Nseg = loss_seg(
                epsilon_raw.to(device),
                preds_Nsigma_C,
                preds_Nseg_C,
                E_tensor,
                poisson_tensor,
            )
            running_seg_vloss += loss_Nseg.item()

        if loss_phi_phase:
            input_seg_C = input_sigma_C[:, :1]
            preds_Nseg_C = Nseg(input_seg_C.to(device))
            loss = alpha_loss_phi * loss_phi(
                epsilon_raw.to(device),
                preds_Nsigma_C,
                preds_Nseg_C,
                E_tensor,
                poisson_tensor,
            )
        else:
            preds_sigma_M = Nsigma(input_sigma_M.to(device))
            loss = loss_M(preds_sigma_M, labels_sigma_M.to(device))
            loss += alpha_loss_mc * loss_mc(preds_Nsigma_C)

        running_vloss += loss.item()

    return running_vloss / (iteration + 1), running_seg_vloss / (iteration + 1)

# Train the Nseg model for one epoch
def train_Nseg_one_epoch():
    Nseg.train()
    running_seg_loss = 0.0
    iteration = 0
    for input_sigma_C, epsilon_raw_C in dataloaders["train_sigma_C"]:
        optimizer_Nseg.zero_grad()
        with torch.no_grad():
            preds_Nsigma_C = Nsigma(input_sigma_C.to(device))
        input_seg_C = input_sigma_C[:, :1]
        preds_Nseg_C = Nseg(input_seg_C.to(device))

        loss_Nseg = loss_seg(
            epsilon_raw_C.to(device),
            preds_Nsigma_C,
            preds_Nseg_C,
            E_tensor,
            poisson_tensor,
        )
        loss_Nseg.backward()
        optimizer_Nseg.step()
        running_seg_loss += loss_Nseg.item()
        iteration += 1

    return running_seg_loss / (iteration + 1)


# Launch the training loop
for epoch in range(EPOCHS):

    if not segmentation_phase:
        # Train Nsigma model
        avg_loss = train_Nsigma_one_epoch()
    else:
        # Train Nseg model
        avg_seg_loss = train_Nseg_one_epoch()

    # Validate models
    with torch.no_grad():
        avg_vloss, avg_seg_vloss = validation_one_epoch()

    if not segmentation_phase:
        myprint(
            f"Epoch [{epoch + 1}/{EPOCHS}] ==> Loss train: {avg_loss} | valid: {avg_vloss}",
            func="train Nsigma",
        )
    else:
        myprint(
            f"Epoch [{epoch + 1}/{EPOCHS}] ==> Loss train: {avg_seg_loss} | valid: {avg_seg_vloss}",
            func="train Nseg",
        )

    # Track and save the best performance of Nsigma
    if not segmentation_phase and avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_epoch = epoch + 1
        if save:
            torch.save(Nsigma.state_dict(), "./models/Nsigma")

    # Track and save the best performance of Nseg
    if segmentation_phase and avg_seg_vloss < best_seg_vloss:
        best_seg_vloss = avg_seg_vloss
        best_epoch = epoch + 1
        if save:
            torch.save(Nseg.state_dict(), "./models/Nseg")

    # Check and update training phases
    if not loss_phi_phase:
        if not segmentation_phase:
            lrs_Nsigma1.step(avg_vloss)
            if lrs_Nsigma1.get_last_lr()[0] != initial_Nsigma_lr:
                lrs_Nsigma2.step(avg_vloss)
                if lrs_Nsigma2.num_bad_epochs == patience_lrs_Nsigma2:
                    segmentation_phase = True
        else:
            lrs_Nseg1.step(avg_seg_loss)
            if lrs_Nseg1.get_last_lr()[0] != initial_Nseg_lr:
                lrs_Nseg2.step(avg_seg_loss)
                if lrs_Nseg2.num_bad_epochs == patience_lrs_Nseg2:
                    loss_phi_phase = True
                    segmentation_phase = False
                    best_vloss = 1e4
