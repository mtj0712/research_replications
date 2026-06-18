import os
import xml.etree.ElementTree as ET
from PIL import Image

from torch.utils.data import Dataset

class ImageNetValDataset(Dataset):
    def __init__(self, val_img_dir, val_annot_dir, class_to_idx, transform=None):
        self.val_img_dir = val_img_dir
        self.val_annot_dir = val_annot_dir
        self.class_to_idx = class_to_idx
        self.transform = transform

        self.img_names = [f for f in os.listdir(val_img_dir) if f.endswith(('.JPEG', '.JPG', '.jpeg', '.jpg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.val_img_dir, img_name)

        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(self.val_annot_dir, xml_name)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        class_name = root.find('object').find('name').text
        label = self.class_to_idx[class_name]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label