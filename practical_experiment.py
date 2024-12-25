import os
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.svm import SVC
from torchvision.datasets import FashionMNIST
from PIL import Image
from smrs import smrs
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

'''
This file is for practical experiment using FashionMNIST dataset.
1. Find representative images using smrs algorithm
2. Save representative images in "output_selected_data" folder
3. Perform classification for representative data using SVM and plot accuracy
'''
def preprocess_data(images):
    # Flatten images to vectors
    flattened = images.view(images.shape[0], -1).numpy()
    pca_model = PCA(n_components=0.99, svd_solver='full')

    flattened = pca_model.fit_transform(flattened)
    return flattened


def save_representatives_by_label1(images, labels, rep_indices, rep_labels, output_dir="output_representatives"):
    #save representative images

    os.makedirs(output_dir, exist_ok=True)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = [idx for idx, rep_label in zip(rep_indices, rep_labels) if rep_label == label]

        if not label_indices:
            print(f"No representatives found for label {label}. Skipping...")
            continue

        print(f"Label {label}: {len(label_indices)} representative images.")
        n_images = len(label_indices)
        fig_width = max(3 * n_images, 5)
        fig_height = 5
        fig, axes = plt.subplots(1, n_images, figsize=(fig_width, fig_height))
        fig.suptitle(f"Representative Images for Label {label}", fontsize=16)

        if n_images == 1:
            axes = [axes]

        for i, idx in enumerate(label_indices):
            img = images[idx].squeeze()
            if img.ndim == 3:
                axes[i].imshow(img.permute(1, 2, 0).numpy())
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

        save_path = os.path.join(output_dir, f"label_{label}_representatives.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved representative images for label {label} to {save_path}")


def save_representatives_by_label(images, labels, indices, rep_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label in np.unique(rep_labels):
        label_dir = os.path.join(output_dir, f"label_{label}")
        os.makedirs(label_dir, exist_ok=True)
        label_indices = indices[rep_labels == label]
        for i, idx in enumerate(label_indices):

            image = images[idx]

            # Convert to PIL Image
            if image.ndim == 2:
                pil_image = Image.fromarray(image.astype(np.uint8))
            elif image.ndim == 3:
                pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
            else:
                raise ValueError("Unsupported image dimension for saving PNG.")

            # Save the image as PNG
            pil_image.save(os.path.join(label_dir, f"image_{i}.png"))


def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    # PCA to reduce dimension.
    pca = PCA(n_components=300, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_flattened)
    X_test_pca = pca.transform(X_test_flattened)

    start = time.time()
    # Train SVM classifier
    svm = SVC(kernel='rbf', C=20, gamma='auto', random_state=42)
    svm.fit(X_train_pca, y_train)

    # Predict and calculate accuracy
    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    end = time.time()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    return accuracy, end-start


# Step 4: Main Function
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_set = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = FashionMNIST(root='./data', train=False, download=True, transform=transform)

    images = train_set.data.numpy()
    labels = train_set.targets.numpy()
    representative_indices = []
    representative_labels = []

    # Process each label independently
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        label_images = train_set.data[label_indices]

        # Preprocess data for this label
        reduced_data = preprocess_data(label_images)

        # select representatives for this label
        repInd, C = smrs(reduced_data, alpha=10)

        # Map representatives back to original indices
        original_rep_indices = label_indices[repInd]
        representative_indices.extend(original_rep_indices)
        representative_labels.extend([label] * len(original_rep_indices))

    representative_indices = np.array(representative_indices)
    representative_labels = np.array(representative_labels)

    # Save representative images grouped by labels
    output_dir = 'output_selected_data/'
    save_representatives_by_label(images, labels, representative_indices, representative_labels, output_dir)

    print(f"Representative indices selected: {len(representative_indices)}")
    # Training data with representative data.
    X_train_rep = train_set.data[representative_indices].numpy()
    y_train_rep = train_set.targets[representative_indices].numpy()

    X_train = X_train_rep/ 255.0
    y_train = y_train_rep

    X_test = test_set.data.numpy() / 255.0
    y_test = test_set.targets.numpy()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # Train and evaluate custom SVM
    test_accuracy,time = train_and_evaluate_svm(X_train, y_train, X_test, y_test)

    print(f"Evaluation Accuracy on Representatives: {test_accuracy * 100:.2f}%")
    print(f"Time measured on Representatives: {time :.2f}Seconds")
