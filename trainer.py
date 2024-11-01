import os
import sys
import sys
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import warnings
import argparse

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from concurrent.futures import ThreadPoolExecutor
from models import *
from utils import ViewDataMO

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dataset_configs = {
    "ROSMAP":{"feature_dims": [200, 200, 200]},
    "LGG":   {"feature_dims": [2000, 2000, 548]},
    "KIPAN": {"feature_dims": [2000, 2000, 445]},
    "BRCA":  {"feature_dims":[1000, 1000, 503]}
}

class Trainer:

    def __init__(self, params):
        self.params = params
        self.view_list = [int(v) for v in params.view_list.split(",")]
        self.classfier_dims = [int(c) for c in params.classifier_dims.split(",")]

        if params.dataset in ['ROSMAP', 'LGG']:
            self.num_class = 2
        elif params.dataset in ['BRCA']:
            self.num_class = 5
        elif params.dataset in ['KIPAN']:
            self.num_class = 3

        self.df_train = pd.read_csv(f"{self.params.dataset}/1_tr.csv").to_numpy()
        self.df_test = pd.read_csv(f"{self.params.dataset}/1_te.csv").to_numpy()
        self.tr_batch_size = self.df_train.shape[0] if self.params.batch_size == -1 else self.params.batch_size
        self.te_batch_size = self.df_test.shape[0]
        # init data loader
        self.train_loader = torch.utils.data.DataLoader(ViewDataMO(self.params.dataset, train=True, view_list=self.view_list), batch_size=self.tr_batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(ViewDataMO(self.params.dataset, train=False, view_list=self.view_list), batch_size=self.te_batch_size, shuffle=False)
        
        # init model
        if len(self.view_list) == 1:
            classifier_dims = [[dataset_configs[self.params.dataset]['feature_dims'][int(v)-1]] + self.classfier_dims for v in self.view_list]
            print(f"[x] building single view classifier, classifier_dims = {classifier_dims}")
            self.model = SV(self.num_class, classifier_dims[0], use_uncertainty=True, loss='digamma')
            self.model_type = "sv"
        elif len(self.view_list) == 2:
            classifier_dims = [[dataset_configs[self.params.dataset]['feature_dims'][int(v)-1]] + self.classfier_dims for v in self.view_list]
            self.model = BI(self.num_class, 2, classifier_dims)
            self.model_type = "bi"
        else:
            classifier_dims = [[dataset_configs[self.params.dataset]['feature_dims'][int(v)-1]] + self.classfier_dims for v in self.view_list]
            self.model = TRI(self.num_class, 3, classifier_dims)
            self.model_type = "tri"
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=1e-5)
        self.model.cuda()

    def train_one_epoch(self, epoch):

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.model_type == 'sv':
                data = Variable(data.cuda())
            else:
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].cuda())

            target = Variable(target.long().cuda())
            self.optimizer.zero_grad()

            if self.model_type == 'sv':
                outputs, loss, u = self.model(data, target, epoch)
                loss.backward()
                self.optimizer.step()

            elif self.model_type == 'bi':
                evidences, evidence_a, loss, u = self.model(data, target, epoch)
                loss.backward()
                self.optimizer.step()

            else:
                evidences, loss = self.model(data, target)
                loss.backward()
                self.optimizer.step()

            if epoch % 50 == 0:
                print(f"[x] training, @ {epoch}, batch @ {batch_idx}, loss = {loss.item()}")

    def test_one_epoch(self, epoch):
        self.model.eval()

        correct_num, data_num = 0, 0
        preds, gts, us = [], [], []
        for batch_idx, (data, target) in enumerate(self.test_loader):
            if self.model_type == 'sv':
                data = Variable(data.cuda())
                with torch.no_grad():
                    target = Variable(target.long().cuda())
                    outputs, loss, u = self.model(data, target, epoch)
                    _, predicted = torch.max(outputs, 1)
                    correct_num += (predicted == target).sum().item()

                    gt = target.cpu().detach().numpy().tolist()
                    predicted = predicted.cpu().detach().numpy().tolist()
                    preds.extend(predicted)
                    gts.extend(gt)
                    us.extend(np.squeeze(u.cpu().detach().numpy()).tolist())

            elif self.model_type == 'bi':
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].cuda())
                data_num += target.size(0)
                with torch.no_grad():
                    target = Variable(target.long().cuda())
                    evidences, evidence_a, loss, u = self.model(data, target, epoch)
                    _, predicted = torch.max(evidence_a.data, 1)
                    correct_num += (predicted == target).sum().item()

                    gt = target.cpu().detach().numpy().tolist()
                    predicted = predicted.cpu().detach().numpy().tolist()
                    preds.extend(predicted)
                    gts.extend(gt)
                    us.extend(np.squeeze(u.cpu().detach().numpy()).tolist())

            else: # tri view
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].cuda())
                data_num += target.size(0)
                with torch.no_grad():
                    target = Variable(target.long().cuda())
                    pred, loss = self.model(data, target)
                    _, predicted = torch.max(pred, 1)
                    correct_num += (predicted == target).sum().item()

                    gt = target.cpu().detach().numpy().tolist()
                    predicted = predicted.cpu().detach().numpy().tolist()
                    preds.extend(predicted)
                    gts.extend(gt)

        if self.params.dataset in ['ROSMAP', 'LGG']:
            acc = accuracy_score(gts, preds)
            measure_1 = f1_score(gts, preds)
            measure_2 = roc_auc_score(gts, preds)
            print(f"epoch = {epoch}, Acc: {acc:0.4f}, F1:  {measure_1:0.4f}, ROC-AUC: {measure_2:0.4f}")
        else:
            acc = accuracy_score(gts, preds)
            measure_1 = f1_score(gts, preds, average='weighted')
            measure_2 = f1_score(gts, preds, average='macro')
            print(f"epoch = {epoch}, Acc: {acc:0.4f}, F1_Weight:  {measure_1:0.4f}, F1_Macro: {measure_2:0.4f}")

        return acc, measure_1, measure_2, gts, preds, us

    def train(self):
        global_acc = 0.
        for epoch in tqdm(range(self.params.epochs)):
            self.train_one_epoch(epoch)
            if epoch % self.params.epochs_val == 0:
                acc, measure_1, measure_2, gts, preds, us = self.test_one_epoch(epoch)
                print(f"test @ epoch {epoch}, acc = {acc}, measure_1 = {measure_1}, measure_2 = {measure_2}")
                if acc > global_acc:
                    global_acc = acc
                    self.save_model()

    def save_model(self):
        if not os.path.isdir(self.params.exp_save_path):
            os.makedirs(self.params.exp_save_path)
        torch.save(self.model.state_dict(), f'{self.params.exp_save_path}/model.pth')

    def load_model(self):
        self.model.load_state_dict(torch.load(f'{self.params.exp_save_path}/model.pth'))

    def write_to_file(self):
        self.load_model()
        acc, measure_1, measure_2, gts, preds, us = self.test_one_epoch(9999)
        f = open(f"{self.params.exp_save_path}/best_results.csv", "w")
        f.write("patient_id,pred,label,u\n")
        if self.model_type in ["sv", "bi"]:
            for i in range(len(preds)):
                f.write(f"{i},{preds[i]},{gts[i]},{us[i]}\n")
        else:
            for i in range(len(preds)):
                f.write(f"{i},{preds[i]},{gts[i]},0\n")
        f.flush()
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=-1, help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train [default: 500]')
    parser.add_argument('--epochs_val', type=int, default=50, help='validation frequency')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--exp_save_path', type=str, default="exp")
    parser.add_argument('--dataset', type=str, default="ROSMAP", choices=["ROSMAP", "LGG", "KIPAN", "BRCA"])
    parser.add_argument('--view_list', type=str, default="1", choices=["1", "2", "3", "1,2", "1,3", "2,3", "1,2,3"])
    parser.add_argument('--classifier_dims', type=str, default="200,200,100")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
    trainer.write_to_file()


