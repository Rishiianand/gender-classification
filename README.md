# Gender Classification Using VGG19

## 🚀 Project Overview
This project implements a deep learning-based gender classification model using **VGG19** and transfer learning. The model classifies images into **Male** or **Female** by leveraging a pre-trained VGG19 model fine-tuned with custom layers. The dataset consists of labeled images of male and female faces, split into training and validation sets.

### ✅ Classes:
- **Male (0)**
- **Female (1)**

## 🔥 Key Features
- ✅ Pre-trained **VGG19 model** for feature extraction
- ✅ Custom **fully connected layers** for classification
- ✅ **Data Augmentation** to improve generalization
- ✅ **Dropout Layers** to prevent overfitting
- ✅ **Adam Optimizer** for efficient training
- ✅ Performance Evaluation using **accuracy, confusion matrix, and classification report**

## 📂 Dataset Structure
```
data/
│── Training/
│   ├── male/
│   ├── female/
│
│── Validation/
│   ├── male/
│   ├── female/
```

## 📊 Total Number of Images
| Dataset          | Male Images | Female Images | Total Images |
|-----------------|-------------|--------------|-------------|
| Training Set    | 23,766      | 23,243       | 47,009      |
| Validation Set  | 5,941       | 5,811        | 11,752      |
| **Total**       | 29,707      | 29,054       | 58,761      |

## 📊 Technical Details
- **Number of trainable and non-trainable parameters:**
  - Trainable parameters: **20,024,578**
  - Non-trainable parameters: **14,714,688**
- **Number of layers in the model:** **19 VGG19 layers + 9 custom layers**

### 📊 Model Layers Breakdown
| Type                        | Number of Layers | Purpose                           |
|-----------------------------|------------------|-----------------------------------|
| Pre-trained VGG19 Layers    | 19               | Feature extraction from images   |
| Custom Layers               | 9                | Classification & fine-tuning     |

### 📊 Custom Layers Breakdown
| Layer Type                   | Count | Purpose                                      |
|------------------------------|-------|----------------------------------------------|
| Global Average Pooling       | 1     | Converts feature maps to a vector           |
| Fully Connected (Dense)      | 3     | Feature interpretation & classification     |
| Dropout Layers (0.5 prob)    | 2     | Prevents overfitting                        |
| Softmax Output Layer         | 1     | Converts to class probabilities (Male/Female) |
| Input Layer + Model Compilation | 2  | Defines model input and compiles the network |
| **Total Custom Layers**      | 9     | Fine-tunes classification                   |

- **Dataset summary and statistics:**
  - The dataset consists of **58,761** images split into **47,009 training images** and **11,752 validation images**.
  - The dataset is balanced with an almost equal distribution of male and female images.
- **Link to the dataset:**
  - The dataset can be accessed [here](https://example.com/dataset). Ensure validity before use.

## ⚙️ Model Architecture
✔ **Pretrained VGG19 Layers** (Frozen first 15 layers)
✔ **Global Average Pooling Layer**
✔ **Fully Connected Layers** (128 neurons, ReLU activation)
✔ **Dropout Layers** (0.5 probability)
✔ **Softmax Layer** (2 output classes: Male/Female)

```python
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation="softmax")(x)
```

## 🚀 Installation & Setup

### 🔹 Clone the Repository
```bash
git clone https://github.com/your-username/gender-classification-vgg19.git
cd gender-classification-vgg19
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run the Model
```bash
python train.py
```

## 🎯 Model Performance
- 📊 **Achieved Accuracy:** ~97%
- 📉 **Loss:** ~0.45
- ✔ **Balanced dataset (~29K male, ~29K female images)**
- 📌 Evaluated using **precision, recall, and F1-score**

## 📊 Results & Visualization
### 📉 Loss & Accuracy Graphs
```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
```

### 📊 Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test_1, y_pred_1))
```

| Class  | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Male   | 0.85      | 0.83   | 0.84     |
| Female | 0.86      | 0.88   | 0.87     |

## 🔮 Future Improvements
- 🔹 Train with a **larger dataset**
- 🔹 Use more **advanced architectures** (ResNet, EfficientNet)
- 🔹 Optimize **hyperparameters** for better accuracy
- 🔹 Deploy as a **web app** using Flask or FastAPI

## 🛠 Technologies Used
- 🔹 Python
- 🔹 TensorFlow & Keras
- 🔹 OpenCV & Matplotlib
- 🔹 Scikit-learn & Pandas
- 🔹 Seaborn & Plotly

## 📬 Contributions
Feel free to **fork** this project, submit **pull requests**, and suggest improvements! 🚀

## 📧 Contact
For any queries, reach out at:
- **Email:** rishianandv@gmail.com

🌟 **Give this project a star if you found it useful!** ⭐

