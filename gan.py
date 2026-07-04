import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
import torch.nn.functional as F
from torchvision import models
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# =========================
# CONFIG
# =========================
IMG_SIZE   = 256
BATCH_SIZE = 4
EPOCHS     = 5
device     = "cuda" if torch.cuda.is_available() else "cpu"

folder_path = "image"
save_path   = "outputs"
os.makedirs(save_path, exist_ok=True)

# =========================
# DATASET
# =========================
class EdgeDataset(Dataset):
    def __init__(self, folder):
        self.files  = os.listdir(folder)
        self.folder = folder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.folder, self.files[idx])).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img).astype(np.float32)
        img = (img / 127.5) - 1
        return img


def get_edges(img):
    img  = ((img + 1) * 127.5).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    edge = edge.astype(np.float32) / 255.0
    edge = edge * 2 - 1
    return edge


# =========================
# GENERATOR
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.d1 = down(1, 64, False)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)

        self.u1 = up(512, 256)
        self.u2 = up(512, 128)
        self.u3 = up(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u1 = self.u1(d4)
        u2 = self.u2(torch.cat([u1, d3], 1))
        u3 = self.u3(torch.cat([u2, d2], 1))
        return self.final(torch.cat([u3, d1], 1))


# =========================
# DISCRIMINATOR
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, A, B):
        return self.net(torch.cat([A, B], dim=1))


# =========================
# VGG PERCEPTUAL EXTRACTOR
# =========================
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=None)
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

vgg_extractor = VGGFeatureExtractor().to(device).eval()


# =========================
# INCEPTION-V3 FOR IS + FID
# =========================
inception_model = inception_v3(
    weights=Inception_V3_Weights.DEFAULT, aux_logits=True
).to(device).eval()
for param in inception_model.parameters():
    param.requires_grad = False


# =========================
# FID FEATURE EXTRACTOR
# =========================
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, inception):
        super().__init__()
        self.inception = inception

    def forward(self, x):
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = self.inception.avgpool(x)
        x = x.flatten(1)
        return x

inception_feat_extractor = InceptionFeatureExtractor(inception_model).to(device).eval()


# =========================
# HELPER: PREPARE FOR INCEPTION
# =========================
def prepare_for_inception(tensor):
    x    = ((tensor + 1) / 2).clamp(0, 1)
    x    = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (x - mean) / std


# =========================
# METRIC FUNCTIONS
# =========================

def compute_psnr(real, fake):
    real_np = ((real + 1) / 2).clamp(0, 1).mul(255).byte().cpu().numpy().astype(np.float64)
    fake_np = ((fake + 1) / 2).clamp(0, 1).mul(255).byte().detach().cpu().numpy().astype(np.float64)
    psnr_vals = []
    for r, f in zip(real_np, fake_np):
        mse = np.mean((r - f) ** 2)
        psnr_vals.append(100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse)))
    return float(np.mean(psnr_vals))


def compute_ssim(real, fake):
    real_np = ((real + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    fake_np = ((fake + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
    ssim_vals = []
    for r, f in zip(real_np, fake_np):
        val = ssim_metric(r, f, channel_axis=2, data_range=1.0)
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))


def compute_rmse(real, fake):
    real_np = ((real + 1) / 2).clamp(0, 1).mul(255).cpu().numpy().astype(np.float64)
    fake_np = ((fake + 1) / 2).clamp(0, 1).mul(255).detach().cpu().numpy().astype(np.float64)
    rmse_vals = [np.sqrt(np.mean((r - f) ** 2)) for r, f in zip(real_np, fake_np)]
    return float(np.mean(rmse_vals))


def extract_inception_features(tensor):
    with torch.no_grad():
        x     = prepare_for_inception(tensor)
        feats = inception_feat_extractor(x)
    return feats.cpu().numpy()


def compute_fid_from_features(real_feats, fake_feats):
    mu_r  = np.mean(real_feats, axis=0)
    mu_f  = np.mean(fake_feats, axis=0)
    sig_r = np.cov(real_feats, rowvar=False) if real_feats.shape[0] > 1 else np.eye(real_feats.shape[1])
    sig_f = np.cov(fake_feats, rowvar=False) if fake_feats.shape[0] > 1 else np.eye(fake_feats.shape[1])
    diff    = mu_r - mu_f
    covmean = sqrtm(sig_r @ sig_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(np.dot(diff, diff) + np.trace(sig_r + sig_f - 2.0 * covmean))
    return fid


def compute_inception_score(fake):
    with torch.no_grad():
        x      = prepare_for_inception(fake)
        logits = inception_model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs   = torch.softmax(logits, dim=1)
        p_y     = probs.mean(dim=0, keepdim=True)
        kl      = probs * (torch.log(probs + 1e-8) - torch.log(p_y + 1e-8))
        kl_mean = kl.sum(dim=1).mean()
        is_score = float(torch.exp(kl_mean).cpu())
    return is_score


def compute_disc_accuracy(pred_real, pred_fake):
    real_correct = (pred_real > 0).float().mean().item()
    fake_correct = (pred_fake < 0).float().mean().item()
    return (real_correct + fake_correct) / 2.0 * 100.0


def compute_edge_consistency(edges, fake):
    edge_np = ((edges + 1) / 2).squeeze(1).detach().cpu().numpy()
    fake_np = ((fake  + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).detach().cpu().numpy()
    scores  = []
    for e, f in zip(edge_np, fake_np):
        f_uint8      = (f * 255).astype(np.uint8)
        gray         = cv2.cvtColor(f_uint8, cv2.COLOR_RGB2GRAY)
        gen_edge     = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        e_bin        = (e > 0).astype(np.float32)
        intersection = (gen_edge * e_bin).sum()
        union        = np.clip(gen_edge + e_bin, 0, 1).sum() + 1e-8
        scores.append(intersection / union)
    return float(np.mean(scores))


def compute_perceptual_loss(real, fake):
    with torch.no_grad():
        r = F.interpolate((real          + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False)
        f = F.interpolate((fake.detach() + 1) / 2, size=(224, 224), mode='bilinear', align_corners=False)
        real_feat = vgg_extractor(r)
        fake_feat = vgg_extractor(f)
    return F.l1_loss(fake_feat, real_feat).item()


def compute_feature_matching_loss(real, fake, discriminator, edges):
    with torch.no_grad():
        inp_real = torch.cat([edges, real],          dim=1)
        inp_fake = torch.cat([edges, fake.detach()], dim=1)
        layers   = list(discriminator.net.children())
        feat_real, feat_fake = inp_real, inp_fake
        for i, layer in enumerate(layers):
            feat_real = layer(feat_real)
            feat_fake = layer(feat_fake)
            if i == 3:
                break
        fm_loss = F.l1_loss(feat_fake, feat_real).item()
    return fm_loss


# =========================
# INIT
# =========================
dataset = EdgeDataset(folder_path)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()

opt_G = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))

g_losses, d_losses = [], []

epoch_psnr         = []
epoch_ssim         = []
epoch_rmse         = []
epoch_fid          = []
epoch_is           = []
epoch_disc_acc     = []
epoch_l1_loss      = []
epoch_perceptual   = []
epoch_edge_consist = []
epoch_feat_match   = []

# Store last batch of each epoch for sample display
last_edges = None
last_imgs  = None
last_fake  = None


# =========================
# SAVE FUNCTION
# =========================
def save_image(tensor, path):
    img = ((tensor + 1) / 2).clamp(0, 1)
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# =========================
# HELPER: TENSOR → NUMPY [0,1]
# =========================
def to_np_rgb(tensor):
    """Convert (C,H,W) tensor in [-1,1] to (H,W,C) numpy in [0,1]."""
    return ((tensor + 1) / 2).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()

def to_np_gray(tensor):
    """Convert (1,H,W) or (H,W) edge tensor in [-1,1] to (H,W) numpy in [0,1]."""
    return ((tensor + 1) / 2).clamp(0, 1).squeeze().detach().cpu().numpy()


# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):

    ep_psnr, ep_ssim, ep_rmse = [], [], []
    ep_is,   ep_dacc          = [], []
    ep_l1,   ep_perc, ep_ec   = [], [], []
    ep_fm                     = []

    all_real_feats = []
    all_fake_feats = []

    for i, imgs in enumerate(loader):

        imgs  = imgs.permute(0, 3, 1, 2).to(device)

        edges = []
        for img in imgs:
            e = get_edges(img.permute(1, 2, 0).cpu().numpy())
            edges.append(e)
        edges = torch.tensor(np.array(edges)).unsqueeze(1).float().to(device)

        fake = G(edges)

        # ---- Train D ----
        opt_D.zero_grad()
        pred_real = D(edges, imgs)
        pred_fake = D(edges, fake.detach())

        loss_real = bce(pred_real, torch.ones_like(pred_real))
        loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))
        d_loss    = (loss_real + loss_fake) * 0.5
        d_loss.backward()
        opt_D.step()

        # ---- Train G ----
        opt_G.zero_grad()
        pred_fake_g = D(edges, fake)
        adv    = bce(pred_fake_g, torch.ones_like(pred_fake_g))
        rec    = l1(fake, imgs)
        g_loss = adv + 50 * rec
        g_loss.backward()
        opt_G.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        # ---- Batch Metrics ----
        with torch.no_grad():
            ep_psnr.append(compute_psnr(imgs, fake))
            ep_ssim.append(compute_ssim(imgs, fake))
            ep_rmse.append(compute_rmse(imgs, fake))
            all_real_feats.append(extract_inception_features(imgs))
            all_fake_feats.append(extract_inception_features(fake))
            ep_is.append(compute_inception_score(fake))
            ep_dacc.append(compute_disc_accuracy(pred_real, pred_fake))
            ep_l1.append(rec.item())
            ep_perc.append(compute_perceptual_loss(imgs, fake))
            ep_ec.append(compute_edge_consistency(edges, fake))
            ep_fm.append(compute_feature_matching_loss(imgs, fake, D, edges))

        if i % 10 == 0:
            save_image(fake[0], f"{save_path}/epoch_{epoch+1}_step_{i}.png")

        # Keep last batch for visualization
        last_edges = edges.detach()
        last_imgs  = imgs.detach()
        last_fake  = fake.detach()

    # ---- FID per epoch ----
    try:
        real_feats_np = np.concatenate(all_real_feats, axis=0)
        fake_feats_np = np.concatenate(all_fake_feats, axis=0)
        fid_val = compute_fid_from_features(real_feats_np, fake_feats_np)
    except Exception as ex:
        print(f"  [FID warning] {ex}")
        fid_val = float('nan')

    epoch_psnr.append(float(np.mean(ep_psnr)))
    epoch_ssim.append(float(np.mean(ep_ssim)))
    epoch_rmse.append(float(np.mean(ep_rmse)))
    epoch_fid.append(fid_val)
    epoch_is.append(float(np.mean(ep_is)))
    epoch_disc_acc.append(float(np.mean(ep_dacc)))
    epoch_l1_loss.append(float(np.mean(ep_l1)))
    epoch_perceptual.append(float(np.mean(ep_perc)))
    epoch_edge_consist.append(float(np.mean(ep_ec)))
    epoch_feat_match.append(float(np.mean(ep_fm)))

    print(f"Epoch {epoch+1:02d} | PSNR={epoch_psnr[-1]:.2f} dB | SSIM={epoch_ssim[-1]:.4f} | "
          f"RMSE={epoch_rmse[-1]:.2f} | FID={epoch_fid[-1]:.4f} | IS={epoch_is[-1]:.4f} | "
          f"D_acc={epoch_disc_acc[-1]:.1f}% | L1={epoch_l1_loss[-1]:.4f} | "
          f"Perc={epoch_perceptual[-1]:.4f} | EdgeC={epoch_edge_consist[-1]:.4f} | "
          f"FM={epoch_feat_match[-1]:.4f}")


# =========================
# FINAL DISPLAY — 4 SAMPLE OUTPUTS
# 4 rows × 3 cols: Edge Input | GAN Output | Real Image
# =========================
N_SAMPLES = 4   # number of samples to show (must be <= BATCH_SIZE)
n_show    = min(N_SAMPLES, last_fake.size(0))

fig0, axes0 = plt.subplots(
    n_show, 3,
    figsize=(13, 4 * n_show),
    num="GAN Sample Outputs — 4 Samples"
)

# Column headers (only on row 0)
col_titles = ["Edge Input", "GAN Output", "Real Image"]
for col, title in enumerate(col_titles):
    axes0[0, col].set_title(title, fontweight='bold', fontsize=20, pad=10)

for row in range(n_show):
    edge_disp = to_np_gray(last_edges[row])          # (H, W)
    gen_disp  = to_np_rgb(last_fake[row])             # (H, W, 3)
    real_disp = to_np_rgb(last_imgs[row])             # (H, W, 3)

    axes0[row, 0].imshow(edge_disp, cmap='gray')
    axes0[row, 1].imshow(gen_disp)
    axes0[row, 2].imshow(real_disp)

    # Row label on the left
    axes0[row, 0].set_ylabel(f"Sample {row + 1}", fontweight='bold',
                              fontsize=16, labelpad=10)

    for col in range(3):
        axes0[row, col].set_xticks([])
        axes0[row, col].set_yticks([])
        for spine in axes0[row, col].spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1.2)

plt.suptitle("GAN Results — Edge to Image Translation",
             fontsize=22, fontweight='bold', y=1.01)
plt.tight_layout()
fig0.savefig('sample_outputs_4.png', dpi=300, bbox_inches='tight')


# =========================
# TRAINING LOSSES PLOT
# =========================
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6), num="Training Losses")

axes1[0].plot(g_losses, color="#215B63", linewidth=1.5, label="Generator Loss")
axes1[0].set_title("Generator Loss",     fontweight='bold')
axes1[0].set_xlabel("Iterations",        fontweight='bold')
axes1[0].set_ylabel("Loss",              fontweight='bold')
axes1[0].legend()

axes1[1].plot(d_losses, color="#2C3947", linewidth=1.5, label="Discriminator Loss")
axes1[1].set_title("Discriminator Loss", fontweight='bold')
axes1[1].set_xlabel("Iterations",        fontweight='bold')
axes1[1].set_ylabel("Loss",              fontweight='bold')
axes1[1].legend()

plt.tight_layout()
fig1.savefig('training_losses.png', dpi=300)


# =========================
# 10 SEPARATE METRIC WINDOWS
# =========================
COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F",
    "#BCBD22", "#17BECF"
]

epochs_x = list(range(1, EPOCHS + 1))

metrics = [
    ("Metric 1 — PSNR",                   "PSNR (Peak Signal-to-Noise Ratio)",             epoch_psnr,         COLORS[0], "PSNR (dB)",       "metric_01_psnr.png"),
    ("Metric 2 — SSIM",                   "SSIM (Structural Similarity Index)",             epoch_ssim,         COLORS[1], "SSIM",            "metric_02_ssim.png"),
    ("Metric 3 — RMSE",                   "RMSE (Root Mean Squared Error)",                 epoch_rmse,         COLORS[2], "RMSE (pixels)",   "metric_03_rmse.png"),
    ("Metric 4 — FID",                    "FID (Fréchet Inception Distance)",               epoch_fid,          COLORS[3], "FID Score",        "metric_04_fid.png"),
    ("Metric 5 — Inception Score",        "IS (Inception Score — Inception-v3)",            epoch_is,           COLORS[4], "IS Score",        "metric_05_is.png"),
    ("Metric 6 — Discriminator Accuracy", "Discriminator Accuracy (%)",                     epoch_disc_acc,     COLORS[5], "Accuracy (%)",    "metric_06_disc_accuracy.png"),
    ("Metric 7 — L1 Pixel Loss",          "L1 Pixel Loss (Reconstruction)",                 epoch_l1_loss,      COLORS[6], "L1 Loss",         "metric_07_l1_loss.png"),
    ("Metric 8 — Perceptual Loss",        "Perceptual Loss (VGG-16 Features)",              epoch_perceptual,   COLORS[7], "Perceptual Loss", "metric_08_perceptual_loss.png"),
    ("Metric 9 — Edge Consistency",       "Edge Consistency (Canny IoU)",                   epoch_edge_consist, COLORS[8], "IoU Score",       "metric_09_edge_consistency.png"),
    ("Metric 10 — Feature Matching Loss", "Feature Matching Loss (Discriminator Features)", epoch_feat_match,   COLORS[9], "FM Loss",         "metric_10_feature_matching.png"),
]

for win_title, metric_name, values, color, ylabel, filename in metrics:
    fig, ax = plt.subplots(figsize=(9, 6), num=win_title)
    ax.plot(epochs_x, values,
            color=color, linewidth=2.5,
            marker='o', markersize=4,
            label=metric_name)
    ax.set_title(metric_name,  fontweight='bold')
    ax.set_xlabel("Epoch",     fontweight='bold')
    ax.set_ylabel(ylabel,      fontweight='bold')
    ax.legend(fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig.savefig(filename, dpi=300)

plt.show()