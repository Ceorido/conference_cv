# ======================================================
# CPU OPTIMIZED FINAL SYSTEM PIPELINE
# Bandwidth Aware Adaptive Resolution CNN
# ======================================================

import torch #library untuk membangun dan melatih CNN
import time #library untuk membuat waktu
import random #library untuk menghitung random time
import numpy as np #library untuk pengolahan data dan hasil eksperimen
import pandas as pd #library untuk pengolahan data dan hasil eksperimen
import torch.nn as nn #library untuk membangun dan melatih CNN
import torch.optim as optim #library untuk membangun dan melatih CNN
import seaborn as sns #visualasi walaupun belum terpakai di training

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

# ======================================================
# CPU SETTINGS
# ======================================================

torch.set_num_threads(8)   # adjust sesuai CPU core

DEVICE = torch.device("cpu")

# ======================================================
# REPRODUCIBILITY
# ======================================================

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ======================================================
# CONFIGURATION
# ======================================================

TRAIN_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_train"
TEST_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_test"
PRED_PATH = r"C:\Users\Lenovo\VisualTrain\projectHaar\dataset\seg_pred"

RESOLUTIONS = [150,112,96,64]

BATCH_SIZE = 8             # lebih stabil CPU

EPOCHS = 8

# ======================================================
# CNN MODEL
# ======================================================

class AdaptiveCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4,4))

        )

        self.classifier = nn.Sequential(

            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,6)

        )

    def forward(self,x):

        x=self.features(x)

        x=x.view(x.size(0),-1)

        x=self.classifier(x)

        return x

# ======================================================
# DATA LOADER PREPROCESSING (CPU SAFE)
# ======================================================

def get_loader(path,res):

    transform = transforms.Compose([

        transforms.Resize((res,res)),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]

        )

    ])

    dataset = datasets.ImageFolder(path,transform)

    loader = DataLoader(

        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,

        num_workers=0,      # CPU WINDOWS FIX
        pin_memory=False    # GPU only feature

    )

    return loader,dataset.classes



# ======================================================
# TRAIN FUNCTION (UNTUK TRACKING LOSS DAN ACCURACY)
# ======================================================

def train(model,train_loader,test_loader):

    criterion=nn.CrossEntropyLoss()

    optimizer=optim.Adam(model.parameters(),lr=0.001)

    best_acc=0

    for epoch in range(EPOCHS):

        model.train()

        loss_total=0
        correct=0
        total=0

        start=time.time()

        for i,(img,label) in enumerate(train_loader):

            img=img.to(DEVICE)
            label=label.to(DEVICE)

            optimizer.zero_grad()

            output=model(img)

            loss=criterion(output,label)

            loss.backward()

            optimizer.step()

            loss_total+=loss.item()

            _,pred=torch.max(output,1)

            correct+=(pred==label).sum().item()

            total+=label.size(0)

            # progress indicator
            if i%50==0:

                print("Batch",i,"/",len(train_loader))

        train_loss=loss_total/len(train_loader)

        train_acc=correct/total

        test_acc,_,_=evaluate(model,test_loader)

        end=time.time()

        print("---------------------")

        print("Epoch:",epoch+1)

        print("Train Loss:",train_loss)

        print("Train Accuracy:",train_acc)

        print("Test Accuracy:",test_acc)

        print("Epoch Time:",end-start)

        if test_acc>best_acc:

            best_acc=test_acc

            torch.save(model.state_dict(),"best_model_cpu.pth")

            print("Best model saved")

# ======================================================
# TEST FUNCTION (Mengukur performa model untuk akurasi, inference time, dan confusion matrix)
# ======================================================

def evaluate(model,loader):

    model.eval()

    correct=0
    total=0

    y_true=[]
    y_pred=[]

    start=time.time()

    with torch.no_grad():

        for img,label in loader:

            img=img.to(DEVICE)

            label=label.to(DEVICE)

            output=model(img)

            _,pred=torch.max(output,1)

            correct+=(pred==label).sum().item()

            total+=label.size(0)

            y_true.extend(label.numpy())

            y_pred.extend(pred.numpy())

    end=time.time()

    acc=correct/total

    infer_time=end-start

    cm=confusion_matrix(y_true,y_pred)

    return acc,infer_time,cm

# ======================================================
# DEPLOYMENT TEST (untuk simulasi performa dan edge computing scenario)
# ======================================================

def deployment_test(model,loader):

    model.eval()

    total=0

    start=time.time()

    with torch.no_grad():

        for img,label in loader:

            img=img.to(DEVICE)

            output=model(img)

            total+=img.size(0)

    end=time.time()

    total_time=end-start

    fps=total/total_time

    return total_time,fps

# ======================================================
# NETWORK SIMULATION
# ======================================================

def bandwidth_sim():

    return random.choice([1,3,5,10])

def latency_calc(res,bw):

    size=res*res*3

    latency=size/(bw*1000)

    return latency

# ======================================================
# ADAPTIVE CONTROLLER (Adaptive Decision System)
# ======================================================

def adaptive_resolution(bw):

    if bw <= 1:
        return 64      # very low bandwidth

    elif bw <= 3:
        return 96      # low bandwidth

    elif bw <= 6:
        return 112     # medium bandwidth

    else:
        return 150     # high bandwidth

# ======================================================
# EXPERIMENT PIPELINE (Bandingkan semua Resolusi Dataset)
# ======================================================

results=[]

for res in RESOLUTIONS:

    print("================================")

    print("Resolution:",res)

    train_loader,_=get_loader(TRAIN_PATH,res)

    test_loader,_=get_loader(TEST_PATH,res)

    pred_loader,_=get_loader(PRED_PATH,res)

    model=AdaptiveCNN().to(DEVICE)

    train(model,train_loader,test_loader)

    acc,inf_time,cm=evaluate(model,test_loader)

    pred_time,fps=deployment_test(model,pred_loader)

    bw=bandwidth_sim()

    latency=latency_calc(res,bw)

    results.append([

        res,
        acc,
        inf_time,
        pred_time,
        fps,
        bw,
        latency

    ])

    print("Final Accuracy:",acc)

    print("FPS:",fps)

    print("Latency:",latency)

# ======================================================
# SAVE RESULTS
# ======================================================

df=pd.DataFrame(

results,

columns=[

"Resolution",
"Accuracy",
"InferenceTime",
"DeploymentTime",
"FPS",
"Bandwidth",
"Latency"

])

df["TradeOff"]=df["Accuracy"]/df["Latency"]

print(df)

df.to_csv("final_system_test_cpu.csv",index=False)

print("Results saved")
