import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------- CONFIG ------------
folder1_path = "train"
folder2_path = "image"

out1_path = "processed"
out2_path = "processed"

IMG_SIZE = 256
MAX_IMAGES = 500

# create output folders
os.makedirs(out1_path, exist_ok=True)
os.makedirs(out2_path, exist_ok=True)


# ----------- LOAD + PREPROCESS FUNCTION ------------
def preprocess_and_save(input_folder, output_folder, max_images=500):
    processed_images = []

    files = sorted(os.listdir(input_folder))[:max_images]

    for i, file in enumerate(files):
        img_path = os.path.join(input_folder, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1️⃣ Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # 2️⃣ Edge Extraction (Canny)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # convert edges to 3 channel (for saving consistency)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # 3️⃣ Normalize (for model use)
        norm_img = edges_3ch / 255.0

        processed_images.append(norm_img)

        # 4️⃣ Save image
        save_path = os.path.join(output_folder, f"img_{i}.png")
        cv2.imwrite(save_path, edges)

    return np.array(processed_images)


# ----------- PROCESS BOTH FOLDERS ------------
data1 = preprocess_and_save(folder1_path, out1_path, MAX_IMAGES)
data2 = preprocess_and_save(folder2_path, out2_path, MAX_IMAGES)

print("Folder1 processed:", data1.shape)
print("Folder2 processed:", data2.shape)


# ----------- DISPLAY 3 SAMPLE IMAGES ------------
def show_samples(images, title):
    plt.figure(figsize=(10, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


show_samples(data1, "Folder 1 Processed (Edges)")
show_samples(data2, "Folder 2 Processed (Edges)")