import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageDraw
from utils import transform
import numpy as np

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder="./", split='TRAIN', keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = "./oracle-detection"
        self.keep_difficult = keep_difficult

        if self.split == "TRAIN":
            label_files = os.listdir(os.path.join(self.data_folder, "train_label"))
        else:
            label_files = os.listdir(os.path.join(self.data_folder, "val_label"))

        image_files = os.listdir(os.path.join(self.data_folder, "img"))
        
        # check the number of file 
        # 训练集有8895个image可以用
        # 测试集有441个image可以用
        self.images = []
        self.objects = []
        for label_file in label_files:
            image_name = label_file.split(".")[0] + '.jpg'
            self.images.append(os.path.join(self.data_folder, "img", image_name))
            with open(os.path.join(self.data_folder, "train_label", label_file), 'r') as j:
                label = json.load(j)
                object = {"boxes":[], "labels":[]}
                ann = label["ann"]
                for i in range(len(ann)):
                    object["boxes"].append(ann[i][:4])
                    object["labels"].append(1)
                self.objects.append(object)         

        assert len(self.images) == len(self.objects)
        print("Datasets length:", len(self.images), " example: ", self.images[0], self.objects[0])

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = labels  # (n_objects)

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each


def draw_image(object):
    boxes = object["boxes"]
    image = Image.open("W00187.jpg")
    draw = ImageDraw.Draw(image)
    
    # 遍历每一个 bounding box 并绘制在图像上
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # 保存带有绘制框的图像
    image.save("test_W00187.jpg")


import cv2
import numpy as np

def draw_bounding_boxes(image_path, bounding_boxes, output_path):
    # 打开图像
    image = cv2.imread(image_path)
    
    # 遍历每一个 bounding box 并绘制在图像上
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        # 绘制矩形框，颜色为红色 (BGR: 0, 0, 255)，线条宽度为2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 保存带有绘制框的图像
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # data = PascalVOCDataset()
    with open(os.path.join("W00187.json"), 'r') as j:
        label = json.load(j)
        object = {"boxes":[], "label":[]}
        ann = label["ann"]
        for i in range(len(ann)):
            object["boxes"].append(ann[i][:4])
            object["label"].append(1)
        draw_image(object)