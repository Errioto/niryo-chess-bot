import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np


class board_data(Dataset):
    def __init__(self, dataset):
        self.X   = [d[0] for d in dataset]
        self.y_p = [d[1] for d in dataset]
        self.y_v = [float(d[2]) for d in dataset]  # Assure-toi que c'est bien un float

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].transpose(2, 0, 1), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*128, 8*8*73)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*128)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self, device=None):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        
class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
def train(net, dataset, epoch_start=0, epoch_stop=1, cpu=0):
    import os
    import datetime
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from alpha_net import AlphaLoss, board_data
    import torch.optim as optim
    import torch

    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)

    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)

    losses_per_epoch = []

    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        losses_per_batch = []

        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if cuda:
                state = state.cuda().float()
                policy = policy.float().cuda()
                value = value.cuda().float()
            else:
                state = state.float()
                policy = policy.float()
                value = value.float()

            optimizer.zero_grad()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses_per_batch.append(loss.item())

        if len(losses_per_batch) == 0:
            print(" Aucun batch n’a été traité à l’epoch", epoch + 1)
            break

        epoch_loss = sum(losses_per_batch) / len(losses_per_batch)
        losses_per_epoch.append(epoch_loss)
        print(f" Epoch {epoch + 1}/{epoch_stop} — Loss: {epoch_loss:.4f}")

        scheduler.step()

    if losses_per_epoch:
        fig = plt.figure()
        ax = fig.add_subplot(222)
        ax.scatter(range(1, len(losses_per_epoch) + 1), losses_per_epoch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Epoch")
        plt.savefig(os.path.join("./model_data/", f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"))
        print(' Finished Training')
    else:
        print(" Aucun entraînement effectué — pas de graphique généré.")
