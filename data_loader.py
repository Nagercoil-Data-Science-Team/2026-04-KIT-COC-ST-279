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


# ----------- LOAD DATA ------------
input_images_1 = load_images(folder1_path, MAX_IMAGES)
input_images_2 = load_images(folder2_path, MAX_IMAGES)


# ----------- SIDE BY SIDE DISPLAY ------------
def show_original_vs_edge(images, title):
    for i in range(3):
        if i >= len(images):
            break

        original = images[i]
        edges = get_edges(original)

        plt.figure(figsize=(8,6))

        # Original
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title(f"Original",fontweight="bold")
        plt.axis("off")

        # Edge
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap="gray")
        plt.title(f"Edge",fontweight="bold")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# ----------- OUTPUT ------------
show_original_vs_edge(input_images_1, "Folder1")
show_original_vs_edge(input_images_2, "Folder2")