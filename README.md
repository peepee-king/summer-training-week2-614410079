# Summer-Training-Week2
# 🧠 RFMiD CNN Classification — Retina Disease Detection with VGG & ResNet

本專案為暑期訓練 Week 2 的作業實作，使用 CNN 模型（VGG16、ResNet18、ResNet50）完成 RFMiD 眼底疾病資料集分類任務，並透過 loss 曲線與訓練準確率分析不同模型表現。使用 wandb 進行訓練監控，並導入 focal loss 解決類別不平衡問題。

---

## 🎯 學習目標

- 理解 VGG 與 ResNet 架構差異（參數量、訓練速度、效果）
- 熟悉圖像分類任務的訓練流程（loss function、optimizer、early stopping）
- 學習 focal loss 解決 class imbalance 問題
- 使用 [Weights & Biases (wandb)](https://wandb.ai/) 可視化訓練過程與超參數實驗結果

---

## 📁 專案結構

```
.
├── rfmid_dataset.py        # RFMiD 資料集定義
├── focal_loss.py           # Focal Loss 實作
├── train_cnn.py            # 主訓練程式
├── README.md
└── Retinal-disease-classification/
    ├── labels.csv          # 影像名稱與疾病標籤
    └── images/             # 影像檔案
```

---

---

## 📊 Dataset: RFMiD (Retinal Fundus Multi-Disease Image Dataset)

- 來源：[Kaggle](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification)
- 影像數量：3200 張眼底圖像
- 分類數：28 種視網膜疾病
- 資料格式：`.csv` 檔中包含圖片名稱與疾病類別對應



## 🧠 支援模型架構

| 模型名稱   | 參數數量 | 特點                         |
|------------|-----------|------------------------------|
| VGG16      | 138M      | 傳統大型卷積網路              |
| ResNet18   | 11M       | 較小但引入 residual block    |
| ResNet50   | 25M       | 更深更準確                   |

模型可自由切換訓練：

```python
model = get_model("resnet18")  # 可選：vgg16、resnet18、resnet50
```

---

## 🛠 使用方法

### 1️⃣ 安裝套件(建議使用conda進行套件版本管理)

```bash
pip install torch torchvision matplotlib pandas scikit-learn wandb
```

登入 wandb：

```bash
wandb login
```

---

### 2️⃣ 準備資料集

# 📁 資料夾結構要求

下載資料後，請將Training set內容放在如下路徑下：

```
./Retinal-disease-classification/
├── labels.csv        # 標註檔：影像名稱與疾病類別(RFMiD_Training_Labels.csv)
└── images/           # 所有眼底影像
```

請確認資料夾名稱與位置正確，否則程式將無法正確讀取！

### 🧾 如何讀取資料集

你可以直接使用我們提供的 `rfmid_dataset.py` 來讀取資料：

```python
from rfmid_dataset import RFMiDDataset

dataset = RFMiDDataset(
    csv_file='Retinal-disease-classification/labels.csv',
    img_dir='Retinal-disease-classification/images/',
    transform=your_transform
)
```

如有需要，也可以自行撰寫 Dataset class，只要能正確回傳 `(image, label)` 即可。

---

---

### 3️⃣ 執行訓練

```bash
python train_cnn.py
```

訓練過程將自動上傳到 wandb 項目中，可視化 loss 與 accuracy 曲線。

---

---

## 🎯 如何使用 Focal Loss（處理類別不平衡）

RFMiD 資料集中可能存在不同疾病類別樣本數差異極大的情況。為了解決這種 class imbalance 問題，我們可以使用 Focal Loss 來加強模型對難分類樣本的學習。

### 🔧 使用步驟如下：

1. **引入 focal_loss.py 中的 FocalLoss 類別**

```python
from focal_loss import FocalLoss
```

2. **初始化 loss function（你可以自行調整 alpha / gamma）**

```python
criterion = FocalLoss(alpha=1.0, gamma=2.0)
```

3. **在訓練時直接使用該損失函數**

```python
loss = criterion(predictions, labels)
```

> 註：`predictions` 為模型輸出的 logits，`labels` 為 ground truth 的 class index

### 📌 參數說明

| 參數      | 功能說明                              |
|-----------|---------------------------------------|
| `alpha`   | 控制正負樣本的平衡，通常設為 1.0 即可 |
| `gamma`   | 抑制容易分類樣本的權重，常設為 2.0    |
| `reduction` | 預設為 'mean'，也可改為 'sum' 或 'none' |

---

你可以將 `focal_loss.py` 放在與 `train_cnn.py` 同層的目錄中，並直接引用使用。

---

## 📈 訓練成果展示（wandb）

> 記得將 wandb 訓練連結附在這裡，例如：
- ResNet18 baseline: [wandb link]
- ResNet18 + FocalLoss: [wandb link]
- VGG16 baseline: [wandb link]

---

## 📌 作業繳交規範

### ✅ 必須完成的項目

1. 使用三種模型（VGG16、ResNet18、ResNet50）訓練 RFMiD 資料集，並記錄 loss 與 accuracy 曲線。
2. 使用 Focal Loss 重新訓練 ResNet18，觀察與原始 CrossEntropyLoss 的差異。
3. 至少嘗試兩種超參數設定（例如不同的 learning rate 或 batch size），請在訓練主程式碼中手動修改並記錄結果。
4. 全程使用 wandb 記錄訓練過程。
5. 提交一份簡單報告總結觀察（含圖表與文字說明）。

### 📂 繳交內容

請提交以下檔案：

- `train_cnn.py`：主訓練程式碼
- `rfmid_dataset.py`：資料集定義
- `focal_loss.py`：Focal Loss 實作
- `result_report.pdf`：訓練結果報告（含 wandb 圖表截圖與簡要說明）
- `wandb_log_link.txt`：你的 wandb 專案公開連結


---


---

## ☁️ GitHub 繳交方式

請每位同學**自行建立一個 GitHub Repository**，將本次作業所有內容上傳，並在報告中附上你的 GitHub Repo 連結。

### ✅ Repository 命名建議

請依照以下格式命名你的個人作業 Repo：

```
summer-training-week2-{學號}
```

例如：

```
summer-training-week2-613410112
```

### 🗂️ Repo 內應包含以下檔案：

```
summer-training-week2-學號/
├── train_cnn.py
├── rfmid_dataset.py
├── focal_loss.py
├── result_report.pdf
├── wandb_log_link.txt
└── README.md
```

### 📌 注意事項

- 請確認你的 GitHub Repo 為 **公開狀態** 或提供我們存取權限。
- 請勿上傳大型模型檔案（例如 .pt 檔）
- `report.pdf` 內容需清楚呈現訓練結果與分析（可含 loss/accuracy 圖表、wandb 截圖）

---
---

