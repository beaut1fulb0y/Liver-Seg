import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import UNetDataset
from model import UNet


# set environment parser
Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=10, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("-e", "--epochs", default=100, type=int, help="training epochs")
Parser.add_argument("-l", "--lr", default=0.0005, type=float, help="learning rate")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")


# general train process
def train(model, dataloader, criterion, optimizer, device, train):
    if train:
        model.train()
        print("training model")
    else:
        model.eval()
        print("testing model")
    running_loss = 0
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_idx, (data, labels) in process_bar:
        data, labels = data.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    return epoch_loss


if __name__ == "__main__":
    args = Parser.parse_args()

    train_dataset = UNetDataset(40, '2d', True)
    test_dataset = UNetDataset(10, '2d', False)

    # Define the sizes of the training, validation, and test sets

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"


    model = UNet(2)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    writer = SummaryWriter()

    best = 100
    best_epoch = -1
    for epoch in range(args.epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device, True)
        test_loss = train(model, test_dataloader, criterion, optimizer,
                                         device, False)

        writer.add_scalar("Training Loss", loss, epoch)
        writer.add_scalar("Test Loss", test_loss, epoch)

        print(f"Epoch: {args.epochs}/{epoch + 1}, Train_loss: {loss}, Test_loss: {test_loss}")
        save_root = args.save_path
        if best > test_loss:
            best_epoch = epoch
            best = test_loss
            torch.save(model.state_dict(), os.path.join(save_root, "best.pth"))
            print(f"best epoch: {epoch + 1}")

        writer.close()

    print(f"best_loss: {test_loss}, best epoch: {best_epoch + 1}")