
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_gaussian_noise(images, variance=0.2):
    noise = torch.randn_like(images) * variance
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)


transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        noisy_images = add_gaussian_noise(images)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


model.eval()
examples = 5

plt.figure(figsize=(10, 6))

with torch.no_grad():
    for i in range(examples):
        image, _ = next(iter(test_loader))
        image = image.to(device)
        noisy_image = add_gaussian_noise(image)

        output = model(noisy_image)

        image = image.cpu().squeeze().numpy()
        noisy_image = noisy_image.cpu().squeeze().numpy()
        output = output.cpu().squeeze().numpy()

        plt.subplot(3, examples, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Oryginalny")

        plt.subplot(3, examples, i + 1 + examples)
        plt.imshow(noisy_image, cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Zaszumiony")

        plt.subplot(3, examples, i + 1 + 2 * examples)
        plt.imshow(output, cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Odszumiony")

plt.tight_layout()
plt.show()
