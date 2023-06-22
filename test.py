import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from skimage import measure
from model import UNet
from dataset import UNetDataset

Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=20, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("--dropout", default=0.1, type=float, help="dropout")
Parser.add_argument("-e", "--epochs", default=100, type=int, help="training epochs")
Parser.add_argument("-l", "--lr", default=0.0005, type=float, help="learning rate")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")

if __name__ == "__main__":
    args = Parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"

    model = UNet(2)
    param_path = os.path.join('params', 'Ubest.pth')
    sd = torch.load(param_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    test_dataset = UNetDataset(11, '2d', False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)[:, 1, :, :]
        biseg = torch.where(out > 0.5, 255, 0)
        biseg = biseg.cpu().numpy()
        for i in range(imgs.shape[0]):
            img = np.uint8(biseg[i])
            labels = measure.label(img, connectivity=2)
            largest_label = np.argmax(np.bincount(labels.flat, weights=img.flat))
            new_img = np.zeros_like(img)
            mask = np.zeros_like(img)
            cv2.circle(mask, (256, 256), 256, (255), thickness=-1)
            new_img[labels == largest_label] = 255
            masked_img = cv2.bitwise_and(new_img, new_img, mask=mask)
            img = Image.fromarray(new_img)
            img.save(os.path.join('runs', f'{i + 40}.png'))
