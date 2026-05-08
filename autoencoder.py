import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from itertools import product
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

KS, ST, PAD = 3, 1, 1
CHANNELS = [3, 15]
LATENT_DIMS = [4, 16, 32]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist(batch_size, train):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def get_balanced_subset(n_per_class=10, batch_size=64):
    full = datasets.MNIST("data", train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
    found_images = {d: [] for d in range(10)}
    found_labels = {d: [] for d in range(10)}
    for img, label in full:
        d = label if isinstance(label, int) else label.item()
        if len(found_images[d]) < n_per_class:
            found_images[d].append(img)
            found_labels[d].append(d)
        if all(len(v) == n_per_class for v in found_images.values()):
            break
    images = torch.cat([torch.stack(found_images[d]) for d in range(10)])
    labels = torch.tensor([l for d in range(10) for l in found_labels[d]])
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


def get_one_per_digit(loader):
    found = {}
    for images, labels in loader:
        for img, label in zip(images, labels):
            d = label.item()
            if d not in found:
                found[d] = img
            if len(found) == 10:
                break
        if len(found) == 10:
            break
    return torch.stack([found[d] for d in range(10)])


def get_n_per_digit(loader, n=5):
    found = {d: [] for d in range(10)}
    for images, labels in loader:
        for img, label in zip(images, labels):
            d = label.item()
            if len(found[d]) < n:
                found[d].append(img)
        if all(len(v) == n for v in found.values()):
            break
    imgs = torch.stack([img for d in range(10) for img in found[d]])
    lbls = torch.tensor([d for d in range(10) for _ in range(n)])
    return imgs, lbls


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self, hidden_channels, latent_dim, kernel_size, stride, padding):
        super().__init__()
        self.kwargs = {
            "hidden_channels": hidden_channels,
            "latent_dim": latent_dim,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
        }
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            after_conv1 = self.conv_encoder[:2](dummy)
            after_conv2 = self.conv_encoder(dummy)
            h1 = after_conv1.shape[2]
            h2 = after_conv2.shape[2]
            self.conv_out_shape = after_conv2.shape[1:]
            self.flat_size = after_conv2.flatten(1).shape[1]

        self.op1 = h1 - ((h2 - 1) * stride - 2 * padding + kernel_size)
        self.op2 = 28 - ((h1 - 1) * stride - 2 * padding + kernel_size)

        self.encoder = nn.Sequential(
            self.conv_encoder,
            nn.Flatten(),
            nn.Linear(self.flat_size, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flat_size),
            nn.ReLU(),
            nn.Unflatten(1, self.conv_out_shape),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=self.op1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, 1,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=self.op2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConvAutoencoderClassifier(ConvAutoencoder):
    def __init__(self, hidden_channels, latent_dim, kernel_size, stride, padding):
        super().__init__(hidden_channels, latent_dim, kernel_size, stride, padding)
        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        return self.classifier(self.encoder(x))


class ConvAutoencoderClassifierTrained(nn.Module):
    """Frozen pretrained autoencoder with a newly trained classifier head."""
    def __init__(self, trained_model: ConvAutoencoder):
        super().__init__()
        self.trained_model = trained_model
        self.kwargs = trained_model.kwargs
        self.classifier = nn.Linear(trained_model.kwargs["latent_dim"], 10)

    def forward(self, x):
        with torch.no_grad():
            z = self.trained_model.encoder(x)
        return self.classifier(z)


class ConvAutoencoderDecoderTrained(nn.Module):
    """Frozen pretrained encoder+classifier with a newly trained decoder head."""
    def __init__(self, trained_model: ConvAutoencoderClassifier):
        super().__init__()
        self.trained_model = trained_model
        self.kwargs = trained_model.kwargs
        kw = trained_model.kwargs
        self.decoder = nn.Sequential(
            nn.Linear(10, trained_model.flat_size),
            nn.ReLU(),
            nn.Unflatten(1, trained_model.conv_out_shape),
            nn.ConvTranspose2d(kw["hidden_channels"], kw["hidden_channels"],
                               kernel_size=kw["kernel_size"], stride=kw["stride"],
                               padding=kw["padding"], output_padding=trained_model.op1),
            nn.ReLU(),
            nn.ConvTranspose2d(kw["hidden_channels"], 1,
                               kernel_size=kw["kernel_size"], stride=kw["stride"],
                               padding=kw["padding"], output_padding=trained_model.op2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        with torch.no_grad():
            z = self.trained_model(x)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reconstruction(model, loader, num_epochs=10, lr=1e-3):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    model.losses = []
    print(f"Training reconstruction  config={model.kwargs}")
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for images, _ in loader:
            images = images.to(device)
            loss = criterion(model(images), images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        median_loss = np.median(batch_losses)
        model.losses.append(median_loss)
        print(f"  Epoch [{epoch+1}/{num_epochs}]  L1={median_loss:.4f}")
    return model


def train_classification(model, loader, num_epochs=10, lr=1e-3):
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.losses = []
    print(f"Training classifier  config={model.kwargs}")
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        median_loss = np.median(batch_losses)
        model.losses.append(median_loss)
        print(f"  Epoch [{epoch+1}/{num_epochs}]  CE={median_loss:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_reconstruction(model, loader):
    device = next(model.parameters()).device
    model.eval()
    total_l1, n_batches = 0.0, 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            total_l1 += nn.functional.l1_loss(model(images), images).item()
            n_batches += 1
    avg_l1 = total_l1 / n_batches
    cfg = model.kwargs
    print(f"  ch={cfg['hidden_channels']} d={cfg['latent_dim']}  L1={avg_l1:.4f}")
    return avg_l1


def evaluate_classification(model, loader):
    device = next(model.parameters()).device
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_ce, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_ce += criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    avg_ce = total_ce / len(loader)
    acc = correct / total
    cfg = model.kwargs
    print(f"  ch={cfg['hidden_channels']} d={cfg['latent_dim']}  CE={avg_ce:.4f}  Acc={acc*100:.1f}%")
    return avg_ce, acc


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_losses(models_list, title):
    fig = go.Figure()
    for m in models_list:
        cfg = m.kwargs
        label = f"ch={cfg['hidden_channels']}, d={cfg['latent_dim']}"
        fig.add_trace(go.Scatter(y=m.losses, mode='lines+markers', name=label))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Median Loss")
    fig.show()


def show_reconstructions(models_dict, images, title="Original vs. Reconstructed"):
    n = images.shape[0]
    rows = 1 + len(models_dict)
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=11, y=1.02)
    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(str(i), fontsize=9)
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[0, 0].axis("on")
    for row, (label, model) in enumerate(models_dict.items(), start=1):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            reconstructed = model(images.to(device)).cpu()
        for i in range(n):
            axes[row, i].imshow(reconstructed[i].squeeze(), cmap="gray")
            axes[row, i].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=8)
        axes[row, 0].axis("on")
    plt.tight_layout()
    plt.show()


def show_multiinstance_reconstructions(models_dict, images, n_per_digit=5,
                                        title="Multi-instance reconstruction"):
    n_digits = 10
    total_cols = n_digits * n_per_digit
    rows = 1 + len(models_dict)
    fig, axes = plt.subplots(rows, total_cols, figsize=(total_cols * 0.9, rows * 1.3))
    fig.suptitle(title, fontsize=11, y=1.02)
    for col in range(total_cols):
        axes[0, col].imshow(images[col].squeeze(), cmap="gray")
        axes[0, col].axis("off")
    for d in range(n_digits):
        axes[0, d * n_per_digit].set_title(str(d), fontsize=8)
    axes[0, 0].set_ylabel("Original", fontsize=8)
    axes[0, 0].axis("on")
    for row, (label, model) in enumerate(models_dict.items(), start=1):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            recon = model(images.to(device)).cpu()
        for col in range(total_cols):
            axes[row, col].imshow(recon[col].squeeze(), cmap="gray")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=7)
        axes[row, 0].axis("on")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_configs():
    return list(product(CHANNELS, LATENT_DIMS))


def main():
    train_loader = get_mnist(batch_size=64, train=True)
    test_loader = get_mnist(batch_size=64, train=False)
    configs = make_configs()

    # --- Type 1: Autoencoders ---
    ae_models = [ConvAutoencoder(c, ld, KS, ST, PAD) for c, ld in configs]
    trained_ae_models = [train_reconstruction(m, train_loader) for m in ae_models]

    # --- Type 2: Encoder+Classifier (full dataset) ---
    cls_models_full = [ConvAutoencoderClassifier(c, ld, KS, ST, PAD) for c, ld in configs]
    trained_cls_models_full = [train_classification(m, train_loader) for m in cls_models_full]

    # --- Type 2: Encoder+Classifier (100-sample balanced subset) ---
    small_loader = get_balanced_subset(n_per_class=10, batch_size=64)
    cls_models_small = [ConvAutoencoderClassifier(c, ld, KS, ST, PAD) for c, ld in configs]
    trained_cls_models_small = [train_classification(m, small_loader) for m in cls_models_small]

    # --- Type 3: Frozen pretrained AE + new Classifier ---
    ae_cls_models = [ConvAutoencoderClassifierTrained(ae) for ae in trained_ae_models]
    trained_ae_cls_models = [train_classification(m, train_loader) for m in ae_cls_models]

    # --- Type 4: Frozen pretrained Encoder+Classifier + new Decoder ---
    cls_dec_models = [ConvAutoencoderDecoderTrained(cls) for cls in trained_cls_models_full]
    trained_cls_dec_models = [train_reconstruction(m, train_loader) for m in cls_dec_models]

    # --- Loss curves ---
    plot_losses(trained_ae_models,        "Type 1 - Autoencoder (L1)")
    plot_losses(trained_cls_models_full,  "Type 2 Full - Encoder+Classifier (CE)")
    plot_losses(trained_cls_models_small, "Type 2 Small - Encoder+Classifier 100-sample (CE)")
    plot_losses(trained_ae_cls_models,    "Type 3 - Pretrained AE + Classifier (CE)")
    plot_losses(trained_cls_dec_models,   "Type 4 - Pretrained Classifier + Decoder (L1)")

    # --- Evaluation ---
    print("=== Type 1: Autoencoder ===")
    for m in trained_ae_models:
        evaluate_reconstruction(m, test_loader)

    print("\n=== Type 2 Full: Encoder+Classifier ===")
    for m in trained_cls_models_full:
        evaluate_classification(m, test_loader)

    print("\n=== Type 2 Small: Encoder+Classifier (100 samples) ===")
    for m in trained_cls_models_small:
        evaluate_classification(m, test_loader)

    print("\n=== Type 3: Pretrained AE + Classifier ===")
    for m in trained_ae_cls_models:
        evaluate_classification(m, test_loader)

    print("\n=== Type 4: Pretrained Classifier + Decoder ===")
    for m in trained_cls_dec_models:
        evaluate_reconstruction(m, test_loader)

    # --- Reconstruction visualizations ---
    vis_loader = get_mnist(batch_size=256, train=True)
    sample_images = get_one_per_digit(vis_loader)

    show_reconstructions(
        {f"AE C={m.kwargs['hidden_channels']} d={m.kwargs['latent_dim']}": m
         for m in trained_ae_models},
        sample_images,
        title="Type 1 - Autoencoder Reconstructions",
    )
    show_reconstructions(
        {f"ClsDec C={m.kwargs['hidden_channels']} d={m.kwargs['latent_dim']}": m
         for m in trained_cls_dec_models},
        sample_images,
        title="Type 4 - Pretrained Classifier + Decoder Reconstructions",
    )

    multi_images, _ = get_n_per_digit(vis_loader, n=5)
    show_multiinstance_reconstructions(
        {"AE (best)": trained_ae_models[2],
         "ClsDec (best)": trained_cls_dec_models[2]},
        multi_images, n_per_digit=4,
        title="Intra-class variability: AE vs Classifier+Decoder (5 instances per digit)",
    )


if __name__ == "__main__":
    main()
