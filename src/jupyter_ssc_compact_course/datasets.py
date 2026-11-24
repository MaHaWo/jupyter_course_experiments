"""Minimal MNIST dataset helpers (handwritten digits 0-9)."""

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Thin wrapper around torchvision's MNIST: 28x28 grayscale images, labels 0-9."""

    def __init__(self, root: str = "data", train: bool = True):
        self.data = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.ToTensor(),  # PIL image -> float tensor in [0, 1], shape (1, 28, 28)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label


def get_dataloaders(root: str = "data", batch_size: int = 64):
    """Return (train_loader, test_loader)."""
    train_loader = DataLoader(MNISTDataset(root, train=True), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MNISTDataset(root, train=False), batch_size=batch_size)
    return train_loader, test_loader
