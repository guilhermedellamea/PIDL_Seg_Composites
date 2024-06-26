import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

seed = 1903

class CustomDataset(Dataset):
    """
    Custom dataset class for handling input data and optional label data.
    
    Args:
        inputs (array-like): Input data.
        labels (array-like, optional): Label data. Default is None.
        transform (callable, optional): Optional transform to be applied to the input data. Default is None.
        target_transform (callable, optional): Optional transform to be applied to the label data. Default is None.
    """

    def __init__(self, inputs, labels=None, transform=None, target_transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]

        transform_seed = random.randint(0, 2**32 - 1)

        random.seed(transform_seed)
        torch.manual_seed(transform_seed)
        if self.transform:
            input_data = self.transform(input_data)
        random.seed(transform_seed)
        torch.manual_seed(transform_seed)
        if self.target_transform:
            label = self.target_transform(label)

        return input_data, label


def get_normalization_functions(dict_data):
    """
    Generate normalization functions based on the training data statistics.

    Args:
        dict_data (dict): Dictionary containing training data.

    Returns:
        tuple: A tuple containing normalization functions for stress output, grayscale image, and Nsigma input.
    """
    # Normalization for stress outputs
    max_stress_value = torch.max(dict_data["train"]["sigma_outputs_M"])

    def norm_stress_output(stress_field):
        return stress_field / max_stress_value

    # Normalization for epsilon values
    all_epsilon = torch.vstack(
        [
            dict_data["train"]["sigma_inputs_M"][:, 1:, :, :],
            dict_data["train"]["sigma_inputs_C"][:, 1:, :, :],
        ]
    )
    epsilon_field_max = torch.max(all_epsilon)
    epsilon_field_min = torch.min(all_epsilon)

    def norm_epsilon(epsilon_field):
        return (epsilon_field - epsilon_field_min) / (
            epsilon_field_max - epsilon_field_min
        ) * 2 - 1

    # Normalization for grayscale images
    all_grayscale_image = torch.vstack(
        [
            dict_data["train"]["sigma_inputs_M"][:, 0, :, :],
            dict_data["train"]["sigma_inputs_C"][:, 0, :, :],
        ]
    )
    grayscale_image_field_max = torch.max(all_grayscale_image)
    grayscale_image_field_min = torch.min(all_grayscale_image)

    def norm_grayscale_image(grayscale_image):
        return (grayscale_image - grayscale_image_field_min) / (
            grayscale_image_field_max - grayscale_image_field_min
        ) * 2 - 1

    def norm_Nsigma_input(Nsigma_input):

        return torch.vstack(
            [
                norm_grayscale_image(Nsigma_input[:1]),
                norm_epsilon(Nsigma_input[1:]),
            ],
        )

    return norm_stress_output, norm_grayscale_image, norm_Nsigma_input


def get_datasets(dict_data):
    """
    Create datasets with normalized and transformed data.

    Args:
        dict_data (dict): Dictionary containing data for training, validation, and testing.

    Returns:
        dict: A dictionary containing datasets for training, validation, and testing.
    """

    norm_stress_output, norm_grayscale_image, norm_Nsigma_input = (
        get_normalization_functions(dict_data)
    )
    transform_crop = v2.RandomCrop(size=128) # Apply random cropping of size 128x128
    transform_Nsigma_input = v2.Compose(
        [
            transform_crop,
            norm_Nsigma_input,
        ]
    )
    transform_Nsigma_output = v2.Compose(
        [
            transform_crop,
            norm_stress_output,
        ]
    )
    transform_epsilon = v2.Compose(
        [
            transform_crop,
        ]
    )
    transform_Nseg_input = v2.Compose(
        [
            transform_crop,
            norm_grayscale_image,
        ]
    )
    ## Creating Datasets
    # train datasets
    dataset_train_sigma_M = CustomDataset(
        dict_data["train"]["sigma_inputs_M"],
        dict_data["train"]["sigma_outputs_M"],
        transform=transform_Nsigma_input,
        target_transform=transform_Nsigma_output,
    )
    dataset_train_sigma_C = CustomDataset(
        dict_data["train"]["sigma_inputs_C"],
        dict_data["train"]["sigma_inputs_C"][:, 1:, :, :],
        transform=transform_Nsigma_input,
        target_transform=transform_epsilon,
    )
    # validation datasets
    dataset_validation_sigma_M = CustomDataset(
        dict_data["validation"]["sigma_inputs_M"],
        dict_data["validation"]["sigma_outputs_M"],
        transform=transform_Nsigma_input,
        target_transform=transform_Nsigma_output,
    )
    dataset_validation_sigma_C = CustomDataset(
        dict_data["validation"]["sigma_inputs_C"],
        dict_data["validation"]["sigma_inputs_C"][:, 1:, :, :],
        transform=transform_Nsigma_input,
        target_transform=transform_epsilon,
    )
    dataset_validation_seg_C = CustomDataset(
        dict_data["validation"]["seg_inputs_C"],
        dict_data["validation"]["seg_outputs_C"],
        transform=transform_Nseg_input,
        target_transform=transform_crop,
    )
    # test datasets
    dataset_test_sigma_C = CustomDataset(
        dict_data["test"]["sigma_inputs_C"],
        dict_data["test"]["sigma_outputs_C"],
        transform=norm_Nsigma_input,
        target_transform=norm_stress_output,
    )
    dataset_test_seg_C = CustomDataset(
        dict_data["test"]["seg_inputs_C"],
        dict_data["test"]["seg_outputs_C"],
        transform=norm_grayscale_image,
        target_transform=None,
    )

    return {
        "dataset_train_sigma_M": dataset_train_sigma_M,
        "dataset_train_sigma_C": dataset_train_sigma_C,
        "dataset_validation_sigma_M": dataset_validation_sigma_M,
        "dataset_validation_sigma_C": dataset_validation_sigma_C,
        "dataset_validation_seg_C": dataset_validation_seg_C,
        "dataset_test_sigma_C": dataset_test_sigma_C,
        "dataset_test_seg_C": dataset_test_seg_C,
    }


def set_seed(seed):
    """
    Set the seed for random number generation to ensure reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    dict_data,
    device,
    batch_size_train=16,
    batch_size_validation=2,
    shuffle=True,
    num_workers=0,
):
    """
    Create dataloaders for training, validation, and testing datasets.

    Args:
        dict_data (dict): Dictionary containing data for training, validation, and testing.
        device (torch.device): Device to be used for data loading.
        batch_size_train (int, optional): Batch size for training. Default is 16.
        batch_size_validation (int, optional): Batch size for validation. Default is 2.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        num_workers (int, optional): Number of worker processes to use for data loading. Default is 0.

    Returns:
        dict: A dictionary containing dataloaders for training, validation, and testing.
    """

    # Fixed seed
    # set_seed(seed)

    # Transform data to tensors
    for ds in dict_data:
        for key in dict_data[ds]:
            dict_data[ds][key] = torch.Tensor(dict_data[ds][key])
    dict_datasets = get_datasets(dict_data)

    pin_memory = True
    # Creating Dataloaders
    dict_dataloaders = {}
    # train dataloaders
    dict_dataloaders["train_sigma_M"] = DataLoader(
        dict_datasets["dataset_train_sigma_M"],
        batch_size=batch_size_train,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device).manual_seed(seed),
    )
    dict_dataloaders["train_sigma_C"] = DataLoader(
        dict_datasets["dataset_train_sigma_C"],
        batch_size=batch_size_train,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device).manual_seed(seed),
    )
    # validation dataloaders
    dict_dataloaders["validation_sigma_M"] = DataLoader(
        dict_datasets["dataset_validation_sigma_M"],
        batch_size=batch_size_validation,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device).manual_seed(seed),
    )
    dict_dataloaders["validation_sigma_C"] = DataLoader(
        dict_datasets["dataset_validation_sigma_C"],
        batch_size=batch_size_validation,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device).manual_seed(seed),
    )
    dict_dataloaders["validation_seg_C"] = DataLoader(
        dict_datasets["dataset_validation_seg_C"],
        batch_size=batch_size_validation,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device).manual_seed(seed),
    )
    # test dataloaders
    dict_dataloaders["test_sigma_C"] = DataLoader(
        dict_datasets["dataset_test_sigma_C"],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dict_dataloaders["test_seg_C"] = DataLoader(
        dict_datasets["dataset_test_seg_C"],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dict_dataloaders

