import glob
import os
from functools import partial

import numpy as np
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms


def transform_image_path(image_path:str,is_val=False,mask_type="colour"):
    if is_val:
        image_path = image_path.replace("pl_preds","gtFine")
        image_path = image_path.replace("_leftImg8bit.png","_gtFine_labelTrainIds.png")
        return image_path
    else:
        path = "data/cityscapes/sam_colour" if mask_type == "colour" else "data/cityscapes/sam"
        basename = image_path.split("/")[-1]
        basename = basename.replace(".png", "_pseudoTrainIds.png")
        return os.path.join(path,basename)

transform_to_val = partial(transform_image_path,is_val=True)

class RefinementDataset(Dataset):
    def __init__(self,data_root="data/cityscapes/",mode="train",mask_type="colour"):
        super().__init__()
        self.pl_paths = glob.glob(os.path.join(data_root,"pl_preds",f"{mode}","**","*.png"),recursive=True)
        self.sam_paths = list(map(partial(transform_image_path,mask_type=mask_type),self.pl_paths))
        self.val_paths = list(map(transform_to_val,self.pl_paths))

        self.transform = transforms.Resize((256,256))

    def __len__(self):
        return len(self.pl_paths)

    def open_image(self, path):
        img = Image.open(path).convert("P")
        img = self.transform(img)

        img = np.array(img)
        img.setflags(write=True)

        return tensor(img).unsqueeze(0)

    def __getitem__(self, index):
        pl_image = self.open_image(self.pl_paths[index])
        sam_image = self.open_image(self.sam_paths[index])
        gt_image  = self.open_image(self.val_paths[index])
        return pl_image, sam_image, gt_image




class GTADataset(Dataset):
    def __init__(self,train_data_root="data/gta/",val_train_root="data/cityscapes/",mode="train",mask_type="colour"):
        super().__init__()
        self.transform_to_sam = partial(self.transform_to_label,mode="val",mask_type=mask_type)
        if mode == "train":
            sam_mode = "sam_colour" if mask_type == "colour" else "sam"
            self.sam_paths = glob.glob(os.path.join(train_data_root,sam_mode,"*_pseudoTrainIds.png"),recursive=True)
            self.labels = list(map(partial(self.transform_to_label,mask_type=mask_type),self.sam_paths))

        elif mode == "val":
            self.labels = glob.glob(os.path.join(val_train_root,"gtFine",f"{mode}","**","*_gtFine_labelTrainIds.png"),recursive=True)
            self.sam_paths = list(map(self.transform_to_sam,self.labels))

        self.transform = transforms.Resize((256,256))

    def transform_to_label(self,path:str,mode="train",mask_type="colour"):
        if mode == "train":
            path = path.replace("sam_colour","labels") if mask_type == "colour" else path.replace("sam","labels")
            label_path = path.replace("_pseudoTrainIds.png","_labelTrainIds.png")
            return label_path
        else :
            image_path = "data/cityscapes/sam_colour" if mask_type == "colour" else "data/cityscapes/sam" 
            basename = path.split("/")[-1]
            basename = basename.replace("_gtFine_labelTrainIds.png", "_leftImg8bit_pseudoTrainIds.png")
            return os.path.join(image_path,basename)

    def __len__(self):
        return len(self.labels)

    def open_image(self, path):
        img = Image.open(path).convert("P")
        img = self.transform(img)

        img = np.array(img)
        img.setflags(write=True)

        return tensor(img).unsqueeze(0)

    def __getitem__(self, index):
        sam_image = self.open_image(self.sam_paths[index])
        gt_image  = self.open_image(self.labels[index])
        return sam_image, gt_image
