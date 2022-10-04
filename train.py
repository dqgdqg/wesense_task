import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from dataset import MyDataset
from model import *
from tqdm import tqdm

import wandb

wandb.init(project="wesense_task", entity="qgding")
print(wandb.run.id)

def train(model, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader)

    train_loss = 0
    cnt = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(pbar):
        pbar.set_description("Train Epoch: {}".format(epoch))
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pbar.set_postfix_str('[{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss += loss.item()
        cnt += 1
    
    train_loss /= cnt
    acc = 100. * correct / len(train_loader.dataset)

    return train_loss, acc


def test(model, data_loader):
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    acc = 100. * correct / len(data_loader.dataset)

    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    return test_loss, acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='For wesense task.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--bn', action='store_true', default=True,
                        help='if bn')
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='if dropout')


    args = parser.parse_args()

    wandb.config = args

    torch.manual_seed(0)

    device = torch.device("cuda")

    train_data = MyDataset('./data/train.csv')
    valid_data = MyDataset('./data/valid.csv')
    test_data = MyDataset('./data/test.csv')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = Net(bn=args.bn, dropout=args.dropout).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_valid_acc = 0.
    for epoch in range(1, args.epochs + 1):
        log = {}
        log['train_loss'], log['train_acc'] = train(model, train_loader, optimizer, epoch)
        log['valid_loss'], log['valid_acc'] = test(model, valid_loader)
        log['test_loss'], log['test_acc'] = test(model, test_loader)
        wandb.log(log)

        if log['valid_acc'] > best_valid_acc:
            best_valid_acc = log['valid_acc']
            torch.save(model.state_dict(), "saved_model/{}_best.pt".format(wandb.run.id))
        # scheduler.step()

    torch.save(model.state_dict(), "saved_model/{}_last.pt".format(wandb.run.id))

if __name__ == '__main__':
    main()