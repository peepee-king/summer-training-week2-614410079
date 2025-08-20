import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from rfmid_dataset import RFMiDDataset
from focal_loss import FocalLoss
from torchvision.models import resnet18
from torchvision.models import vgg16
from torchvision.models import resnet50
from datetime import datetime
import wandb
import time

train_split=0.8
patience=3
wandb.init(project="summer_hw2", name=f"resnet18_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",dir="week2_wandb", config={"epochs":15,"learning_rate":0.0001,"batch_size":32})
config = wandb.config
lr=config.learning_rate
num_epochs=config.epochs
batch_size=config.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#轉換為正態分布(隨機抽樣計算得到)
])

dataset = RFMiDDataset(
    csv_file='./Retinal-disease-classification/RFMiD_Training_Labels.csv',
    img_dir='./Retinal-disease-classification/images/',
    transform=transform
)

train_dataset,val_dataset=random_split(dataset,[train_split,1-train_split],torch.Generator().manual_seed(0))#讓每次分割結果都一樣
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

model = resnet18(pretrained=True)
# in_features=model.classifier[6].in_features
# model.classifier[6]=nn.Linear(in_features,2)
feature=model.fc.in_features#原始output特徵數 #resnet
model.fc=nn.Linear(feature,2)
model.to(device)

# criterion=nn.CrossEntropyLoss()
criterion=FocalLoss(alpha=1.0,gamma=2.0)
optimizer=Adam(model.parameters(),lr=lr)
best_loss=float('inf')

for epoch in range(num_epochs):
    start_time=time.time()
    model.train()
    total_correct=0
    total=0
    train_loss = 0.0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        predictions=model(images)
        loss=criterion(predictions,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    end_time=time.time()
    train_time=end_time-start_time
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader: 
            images=images.to(device)
            labels=labels.to(device)
            predictions=model(images)
            loss=criterion(predictions,labels)
            val_loss += loss.item()

            _,predicted=torch.max(predictions,1)
            total_correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
        
        accuracy=total_correct/total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        m,s=divmod(train_time,60)
        wandb.log({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss, 
            "accuracy": accuracy,
            "train_time":f"{m:.0f}m{s:.0f}s",
            "epoch":epoch+1
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, Train Time: {m:.0f}m{s:.0f}s")
    if val_loss<best_loss:
        best_loss=val_loss
        trigger_time=0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        trigger_time+=1
        if trigger_time>=patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
# torch.save(model.state_dict(), "model.pt")


