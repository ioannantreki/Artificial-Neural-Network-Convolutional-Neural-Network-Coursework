# Artificial-Neural-Network-Convolutional-Neural-Network-Coursework
Academic Machine Learning coursework focusing on two primary models: an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN). Includes model evaluation using the [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset. 

## 📚 Project Overview
- Preprocess the MNIST dataset.
- Train an ANN and a CNN model to classify handwritten digits.
- Evaluate the models using accuracy and confusion matrices.
- Visualize the training and validation accuracy/loss for analysis.

## 📊 Dataset
The MNIST dataset consists of grayscale images of size 28x28 pixels, representing handwritten digits from 0 to 9. It includes:

- **Training Set**: 60,000 images.
- **Test Set**: 10,000 images.

### Dataset preprocessing steps:
- Normalize pixel values to the range [0, 1].
- Reshape images for the CNN model.

## ✨Features
1. **Artificial Neural Network (ANN)**:
- Input layer: Flattened 28x28 images.
- Two hidden layers with 256 and 128 units (ReLU activation).
- Output layer with 10 units (softmax activation).

2. **Convolutional Neural Network (CNN)**:
- Two convolutional layers with ReLU activation.
- Max pooling for feature reduction.
- Dense layers for classification.

3. **Metrics**:
- Training and validation **accuracy/loss**.
- **Confusion matrix** for performance analysis.

## 📚 Results
1. **ANN**:
**Test accuracy**: Achieved ~0.98% after 10 epochs.
**Confusion matrix provided for detailed analysis*.

2. **CNN**:
**Test accuracy**: Achieved ~0.99% after 10 epochs.
**Confusion matrix provided for detailed analysis*.

**Graphs of training/validation accuracy and loss are available for both models.*

## 🚀 How to Use
1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/mnist-classification.git
cd mnist-classification
```
2. **Install dependencies**:
```bash
pip install numpy tensorflow matplotlib scikit-learn
```
3. **Run the script**:
```bash
python mnist_classification.py
```
## ℹ️ Contact
For any inquiries or collaboration requests, please reach out via GitHub or email at ioannadreki31@gmail.com.
