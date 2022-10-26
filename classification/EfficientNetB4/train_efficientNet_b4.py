
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# import Dataloader, ImageFolder from torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import wandb
import torchvision
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import efficientnet

# from tqdm import tqdm
from tqdm import tqdm

device = torch.device("cuda")

import torch.nn as nn
class Model_classify(nn.Module):
    def __init__(self, model_type):
        super(Model_classify, self).__init__()
        if model_type == 'efficientNet_b4':
            self.model_backbone = models.efficientnet_b4(pretrained=True)
        elif model_type == 'efficientNet_b5':
            self.model_backbone = models.efficientnet_b5(pretrained=True)
        elif model_type == 'efficientNet_b6':
            self.model_backbone = models.efficientnet_b6(pretrained=True)
        elif model_type == 'efficientNet_b7':
            self.model_backbone = models.efficientnet_b7(pretrained=True)
        
        self.sequence_model = nn.Sequential( nn.Linear(1000, 512),
                                                nn.Mish(),
                                                nn.Dropout(0.3),
                                                nn.Linear(512, 2)
                                            )
    def forward(self, x):
        x = self.model_backbone(x)
        x = self.sequence_model(x)
        return x
    
# import f1_score, recall, precision from sklearn.metrics
from sklearn.metrics import f1_score, recall_score, precision_score


class Run_model():
    def __init__(self, model_classify, optimizer, criterion, dict_dataloader, scheduler):
        self.model_classify = model_classify.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = dict_dataloader['train_loader']
        self.val_loader = dict_dataloader['val_loader']
        self.test_loader = dict_dataloader['test_loader']
        # self.scheduler = scheduler
        self.best_f1 = 0
    def train(self, epoch):
        print("Training in epoch {}".format(epoch))
        self.model_classify.train()
        total_loss = 0
        preds, targets = [], []
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data = data.to(device)
            target = target.to(device)
            self.optimizer.zero_grad()
            output = self.model_classify(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()


            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

            preds += predicted.tolist()
            targets += target.tolist()
        # scheduler.step()


        total_loss = round(total_loss / len(self.train_loader), 3)
        print("Training loss: {}".format(total_loss))

        f1 = f1_score(targets, preds, average='macro')
        f1 = round(f1, 3)
        recall = recall_score(targets, preds, average='macro')
        recall = round(recall, 3)
        precision = precision_score(targets, preds, average='macro')
        precision = round(precision, 3)

        print("f1: {}, recall: {}, precision: {}".format(f1, recall, precision))
        # log to wandb
        wandb.log({"train_loss": total_loss,
                   "train_f1": f1, 
                   "train_recall": recall,
                   "train_precision": precision
                   # log the learning rate from the scheduler
                    #  "learning_rate": self.scheduler.get_last_lr()[0]
                })
                   

    def validate(self, epoch):
        print("Validating in epoch {}".format(epoch))
        self.model_classify.eval()
        total_loss = 0
        preds, targets = [], []
        for batch_idx, (data, target) in enumerate(tqdm(self.val_loader)):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                output = self.model_classify(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                preds += predicted.tolist()
                targets += target.tolist()
        total_loss = round(total_loss / len(self.val_loader), 3)
        print("Validation loss: {}".format(total_loss))
        
        f1 = f1_score(targets, preds, average='macro')
        f1 = round(f1, 3)
        recall = recall_score(targets, preds, average='macro')
        recall = round(recall, 3)
        precision = precision_score(targets, preds, average='macro')
        precision = round(precision, 3)

        print("f1: {}, recall: {}, precision: {}".format(f1, recall, precision))

        # log to wandb
        wandb.log({"val_loss": total_loss,
                     "val_f1": f1, 
                     "val_recall": recall,
                     "val_precision": precision})

        if f1 > self.best_f1 and f1 > 0.9:
            self.best_f1 = f1
            torch.save(self.model_classify.state_dict(), "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/weight/checkpoints_efficientNetB4/{}_f1_{}.pt".format(args.model_type, f1))
            #save model in jit file 
            torch.jit.save(torch.jit.script(self.model_classify), "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/weight/checkpoints_efficientNetB4/{}_jtit_f1_{}.pt".format(args.model_type, f1))
            print("Saved best model")
        


    def test(self, epoch):
        print("Testing in epoch {}".format(epoch))
        self.model_classify.eval()
        total_loss = 0
        preds, targets = [], []
        for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
            data = data.to(device)
            target = target.to(device)
            
            with torch.no_grad():
                output = self.model_classify(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                preds += predicted.tolist()
                targets += target.tolist()
        total_loss = round(total_loss / len(self.test_loader), 3)
        print("Testing loss: {}".format(total_loss))
        f1 = f1_score(targets, preds, average='macro')
        f1 = round(f1, 3)
        recall = recall_score(targets, preds, average='macro')
        recall = round(recall, 3)
        precision = precision_score(targets, preds, average='macro')
        precision = round(precision, 3)

        print("f1: {}, recall: {}, precision: {}".format(f1, recall, precision))

        # log to wandb
        wandb.log({"test_loss": total_loss,
                     "test_f1": f1, 
                     "test_recall": recall,
                     "test_precision": precision})
                    


import numpy as np
def config_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

import argparse
from tkinter import S
def input_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--padding_name', type=str, default="same")
    parser.add_argument('--model_type', type=str, default="____")
    args = parser.parse_args()
    return args

def get_dataloader(root_data_dict, batch_size):
    train_path = root_data_dict['train_path']
    val_path = root_data_dict['val_path']
    test_path = root_data_dict['test_path']
    # normalize ImageNet mean and std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # define transforms for the training, validation, and testing   
    train_transforms = [transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        normalize]
    
    val_transforms = [transforms.Resize((224,224)),
                      transforms.ToTensor(),
                      normalize]
    
    test_transforms = [transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        normalize]
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transforms.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform = transforms.Compose(val_transforms))
    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform = transforms.Compose(test_transforms))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    
    # test_loader = 0
    
    dict_dataloader = {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader}
    return dict_dataloader


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return loss.mean() if self.reduction == 'mean' else loss.sum()




if __name__ == '__main__':
    args = input_params()
    config_seed(args.seed)

    print("RUNNING...............................")
    # get dataloader
    batch_size = args.batch_size
    epochs = args.epochs


    train_path = "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/data_png/train"
    val_path =   "/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/data_png/val"
    test_path = '/workspace/nabang1010/STEAM/LeGiaBach_STEAM/DATA/data_png/test'
    root_data_dict = {'train_path': train_path, 'val_path': val_path, 'test_path': test_path}
    dict_dataloader = get_dataloader(root_data_dict, batch_size)


    # get model
    model_classify = Model_classify(args.model_type)
    model_classify = model_classify.to(device)
    # model_classify = nn.DataParallel(model_classify, device_ids = [0,1,2,3])
    # use folcal loss criterion
    # criterion = FocalLoss(gamma=2)
    criterion = nn.CrossEntropyLoss()
    

    # def __init__(self, model_classify, optimizer, criterion, dict_dataloader):

    # use Adam optimizer
    optimizer = torch.optim.Adam(model_classify.parameters(), lr=args.lr, weight_decay=args.wd)
    # use optimizer scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    scheduler = None
    


    # init wandb
    wandb.init(project="STEAM", group = "LeGiaBach_EfficientNetB4", name = "{}_{}_{}".format(args.padding_name, args.lr, args.wd))
    wandb.watch(model_classify)
    # wandb.watch(optimizer)
    #
    run_model = Run_model(model_classify = model_classify, optimizer = optimizer, criterion = criterion, dict_dataloader = dict_dataloader, scheduler = scheduler)
    
    for epoch in range(epochs):
        run_model.train(epoch)
        run_model.validate(epoch)
        run_model.test(epoch)











