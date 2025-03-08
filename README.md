# Skin Cancer Binary Classifier

This repository contains a deep learning-based binary classifier for detecting skin cancer using CNN architectures like **ResNet50** and **DenseNet121**.  

## Dataset Links  

[Original dataset](https://d4h9zj04.na1.hs-sales-engage.com/Ctc/L1+23284/d4H9ZJ04/JlF2-6qcW8wLKSR6lZ3q4V5X7nm6KjQpcW8znnn58l8tL8W6Z7vJs77kHCTW5VWxff4z9182W5RHlXW5Dbj3JW2hdT7B2hCN44W88-KsZ2-C8l7W5fF5Fz61JG_KVxtBHr4wwydbW5PQ1W46mZT77N1JCcXKM9vzZW2-ZG8956_dnlN10s68GV9JJjW2Dtmmy3RnDCSW5KvHXw2NtYWrW2-cD2n5tYfzxVfcr1s28btp6Vrvsv46Zvd_WW2bNBhf8r81cYVQRgHk7XhxDsW2gsV7z38pN3dW8GP86B8dH_cqW6J7VP75gbR5-W93cDQ781jRgpW8fmNBw1-dDt_W6SbRfy2lmdPGVFMjdk7N7bd7W8GWCkT6_DjtCf7DrSj-04)

[Updated train dataset](https://drive.google.com/drive/folders/1Ncs4iVggUj6jYuzvpCHKJD9G_KPy6i3A?usp=sharing)

[Updated test dataset](https://drive.google.com/drive/folders/1ue9FWiTtxMWSal0eLehFYK5Lh5vtLI8N?usp=drive_link)

[Fine-tuned models](https://drive.google.com/drive/folders/1a0vIs98jVATxSJ8FrqTujBRrWIZunzsF?usp=drive_link)

## Installation  

After creating a virtual environment, install the required dependencies:  
```
pip install -r requirements.txt
```

## Steps Followed in the notebook

### 1️⃣ Data Loading  
- Load the original dataset into memory for processing.

### 2️⃣ Data Preprocessing  
Improve dataset quality by detecting and correcting common issues:  

- **Blurry Images:** Identified using **Laplacian Variance**.  
  - Low variance images are sharpened and saved in the `sharpened` folder.  
- **Sharp Images:** Outliers in blurriness distribution are removed.  
- **Dark Images:** Low-brightness images are enhanced using **contrast adjustment** and **histogram equalization** (`brightened` folder).  
- **Color Transformation:** Standardizing RGB values between 0 and 1 to maintain consistency across different lighting conditions (`color_trans` folder).  

### 3️⃣ Custom DataLoader  
- Converts each image into a **tensor**.  
- Normalizes the image using **mean and standard deviation**.  
- Loads the corresponding **labels**.  
- Handles **batching, shuffling, and efficient loading**.  
- **Use `color_trans` dataset** for faster processing.  

### 4️⃣ Model Selection  
We evaluate several CNN-based models and choose the best performers:  
- ✅ **ResNet50** (with **SGD optimizer**)  
- ✅ **DenseNet121**  

### 5️⃣ Model Training  

#### 🔹 ResNet50 - Custom Classifier  
Modified classification block for better generalization:  
```python
Sequential(
  (0): Linear(in_features=2048, out_features=1024, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.6, inplace=False)
  (3): Linear(in_features=1024, out_features=512, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.6, inplace=False)
  (6): Linear(in_features=512, out_features=2, bias=True)
)
```
**Fine-Tuned Models:**  
- Adam Optimizer → `models/ResNetModel_Mar7_best_model.pth`  
- SGD Optimizer → `models/ResNetModel_Mar7_SGD_best_model.pth`  

#### 🔹 DenseNet121 - Custom Classifier  
Modified classification block:  
```python
Sequential(
  (0): Linear(in_features=1024, out_features=512, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.5, inplace=False)
  (3): Linear(in_features=512, out_features=2, bias=True)
  (4): Softmax(dim=1)
)
```
**Fine-Tuned Model:**  
- `models/DenseNetModel_Mar7_best_model.pth`

update `checkpoint_path` variable to run a quick evaluation

---

## 🎯 Model Performance  

| Model                  | Optimizer | Train Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
|------------------------|-----------|----------------|----------------------|---------------|-----------|
| **ResNet50**          | Adam      | 80.75%         | 73.01%               | 52.20%        | 0.9361    |
| **ResNet50**          | SGD       | 92.60%         | 89.60%               | 90.35%        | 0.2535    |
| **DenseNet121**       | Adam      | 92.15%         | 85.73%               | 92.65%        | 0.3839    |

---

## Post-processing  

Adjusting the classification **threshold** helps optimize performance based on the specific use case:  

- **Lower False Alarms** → Set a **higher threshold** to ensure only high-confidence predictions are accepted.  
- **Lower Misses** → Set a **lower threshold** to minimize false negatives, capturing more positive cases.  

Fine-tune the **threshold** to strike the right balance between **precision and recall** based on application needs.  

## Conclusion  

The **DenseNet121 model outperformed ResNet50**, achieving an **F1 Score of 92.65%** compared to **ResNet50's 90.22%**.  
This is expected since **DenseNet121** has a **deeper architecture** and **better feature propagation**, leading to superior performance.  

---

## Acknowledgment  

This project is built using **PyTorch**, **OpenCV**, **Torchvision**, and various deep learning techniques.  

