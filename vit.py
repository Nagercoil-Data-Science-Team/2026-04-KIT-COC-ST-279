import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# ----------- CONFIG ------------
folder1_path = "image"
folder2_path = "train"

IMG_SIZE = 256
PATCH_SIZE = 16
MAX_IMAGES = 500


# ----------- LOAD FUNCTION ------------
def load_images(folder_path, max_images=500):
    images = []

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return np.array([])

    files = sorted(os.listdir(folder_path))[:max_images]

    for file in files:
        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        images.append(img)

    return np.array(images)


# ----------- EDGE EXTRACTION ------------
def get_edges(image):
    img_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


# ----------- 🔵 VISION TRANSFORMER (SIMPLIFIED PATCH FEATURE EXTRACTION) ------------
def vit_feature_extraction(image, patch_size=16):
    """
    Simulated ViT:
    - Split image into patches
    - Compute mean intensity per patch
    - Build feature map
    """

    h, w, c = image.shape
    feature_map = []

    for i in range(0, h, patch_size):
        row = []
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            mean_val = np.mean(patch)
            row.append(mean_val)
        feature_map.append(row)

    return np.array(feature_map)


# ----------- LOAD DATA ------------
input_images_1 = load_images(folder1_path, MAX_IMAGES)
input_images_2 = load_images(folder2_path, MAX_IMAGES)


# ----------- VISUALIZATION FUNCTION ------------
def show_full_pipeline(images, title):
    for i in range(3):
        if i >= len(images):
            break

        img = images[i]

        # Preprocessing
        edges = get_edges(img)

        # ViT feature map
        vit_map = vit_feature_extraction(img)

        # -------- PLOT ----------
        plt.figure(figsize=(12,4))

        # Original
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title(f" Original",fontweight="bold")
        plt.axis("off")

        # Edge
        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap="gray")
        plt.title("Edge Map",fontweight="bold")
        plt.axis("off")

        # ViT Feature Map
        plt.subplot(1, 3, 3)
        plt.imshow(vit_map, cmap="viridis")
        plt.title("ViT Feature Map",fontweight="bold")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# ----------- OUTPUT ------------
show_full_pipeline(input_images_1, "Folder1")
show_full_pipeline(input_images_2, "Folder2")