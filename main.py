import torch
import torch as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import timm
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from src.dataset import DatasetImmobilierSoCal
from src.model import ImageClassifier
from src.trainer import ModelTrainer


def main():


    df_full = pd.read_csv('data/socal2.csv')

    def calculate_label(prix):
        if prix >= 1000000: return 0
        elif prix <= 400000: return 1
        else: return 2

    df_full['label'] = df_full['price'].apply(calculate_label)


    mes_transformations = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    df_train, df_test = train_test_split(
    df_full,
    test_size=0.2,
    random_state=42,
    stratify=df_full['label']
)

    print(f"Houses for training : {len(df_train)}")
    print(f"Houses for test : {len(df_test)}")


    dataset_train = DatasetImmobilierSoCal(dataframe_deja_pret=df_train,
                                        img_dir='data/socal2/socal_pics',
                                        transform=mes_transformations)
    dataset_test = DatasetImmobilierSoCal(dataframe_deja_pret=df_test,
                                        img_dir='data/socal2/socal_pics',
                                        transform=mes_transformations)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)


    print("\n--- 2. Tools and Model Creation... ---")


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ImageClassifier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n -- 3. Launch Training --")

    trainer = ModelTrainer(model, optimizer, criterion, device)

    trainer.train(dataloader_train, epochs=10)

if __name__ == "__main__":
    main()