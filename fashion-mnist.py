from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import time

'''
This file is to measure classification accuracy with whole FashionMNIST dataset.
1. Load FashionMNIST dataset
2. Perform classification using SVM and plot accuracy
'''

transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))  # Normalizes to [-1, 1]
])

#Load FashionMNIST dataset
train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Convert PyTorch dataset to NumPy arrays
X_train_full = train_dataset.data.numpy()
y_train_full = train_dataset.targets.numpy()
X_test_full = test_dataset.data.numpy()
y_test_full = test_dataset.targets.numpy()

# Normalize and select samples for training
X_train = X_train_full / 255.0
y_train = y_train_full
X_test = X_test_full / 255.0
y_test = y_test_full

# Flatten images for PCA
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train_flattened)
X_test_pca = pca.transform(X_test_flattened)

start = time.time()
# Train SVM classifier
svm = SVC(kernel='rbf',C=20, gamma='auto', random_state=42)
svm.fit(X_train_pca, y_train)

# Predict and calculate accuracy
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
end = time.time()
time = end - start

print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

print(f"Time measured: {time :.2f}Seconds")
