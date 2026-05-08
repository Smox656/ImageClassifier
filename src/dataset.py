
import os
from PIL import Image
from torch.utils.data import Dataset

class DatasetImmobilierSoCal(Dataset):

    def __init__(self, dataframe_deja_pret, img_dir, transform=None):
        self.dataframe = dataframe_deja_pret.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        nom_image = str(self.dataframe['image_id'].iloc[idx])
        

        if not nom_image.endswith('.jpg'):
            nom_image += '.jpg'

        chemin_image = os.path.join(self.img_dir, nom_image)
        

        image = Image.open(chemin_image).convert("RGB")


        prix = self.dataframe['price'].iloc[idx]

        if prix >= 1000000:
            label = 0  
        elif prix <= 400000:
            label = 1  
        else:
            label = 2  


        if self.transform:
            image = self.transform(image)

        return image, label