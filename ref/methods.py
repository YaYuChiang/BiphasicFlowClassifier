import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib

import torch
import torch.nn as nn
import torch.optim as optim

class CNet():
    def __init__(self):
        self.model = None

    def fit(self, data, label):
        self.model.fit(data, label)
        return self.model.predict(data)
    
    def predict(self, pred_data):
        return self.model.predict(pred_data)

class LR(CNet):
    def __init__(self, exp_num):
        super(LR, self).__init__()
        if exp_num == 1:
            self.model = LogisticRegression(solver='lbfgs', max_iter=5000)
        elif exp_num == 2:
            self.model = LogisticRegression(C=1e-2, solver='lbfgs', max_iter=100)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class SVMC(CNet):
    def __init__(self, exp_num):
        super(SVMC, self).__init__()
        if exp_num == 1:
            self.model = SVC(kernel='rbf', C=1.0, probability=True)
        elif exp_num == 2:
            self.model = SVC(kernel='poly', C=1e-2, probability=True, gamma=10)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class RF(CNet):
    def __init__(self, exp_num):
        super(RF, self).__init__()
        if exp_num == 1:
            self.model = RandomForestRegressor()
        elif exp_num == 2:
            self.model = RandomForestClassifier(max_depth=2, min_samples_split=100, min_samples_leaf=100)
    
    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class XGB(CNet):
    def __init__(self, exp_num):
        super(XGB, self).__init__()
        if exp_num == 1:
            self.model = XGBClassifier(n_estimators=100, learning_rate=0.02)
        elif exp_num == 2:
            self.model = XGBClassifier(n_estimators=100, learning_rate=0.02)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class GB(CNet):
    def __init__(self, exp_num):
        super(GB, self).__init__()
        if exp_num == 1:
            self.model = GradientBoostingRegressor(n_estimators=100, min_samples_split=10, min_samples_leaf=10)
        elif exp_num == 2:
            self.model = GradientBoostingClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=10)
    
    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class DT(CNet):
    def __init__(self, exp_num):
        super(DT, self).__init__()
        if exp_num == 1:
            self.model = DecisionTreeClassifier()
        elif exp_num == 2:
            self.model = DecisionTreeClassifier()

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class MLP_Model(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
class MLP(CNet):
    def __init__(self, input_dim, epochs, batch_size):
        self.model = MLP_Model(input_dim).cuda()
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.000005)
        self.log_path = "../record/MLP.log"
        # self.model_path = "../record/MLP.pth"
        log_file = open(self.log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()


    def fit(self, data, label):
        self.model.train()
        for e in range(self.epochs):
            total_loss, total_num = 0.0, 0
            index = np.arange(len(data))
            np.random.shuffle(index)
            bar = tqdm(range(len(index)//self.batch_size))
            for iteration in bar:
                self.optimizer.zero_grad()
                data_batch = torch.Tensor(data[index[iteration*self.batch_size:(iteration+1)*self.batch_size]]).cuda()
                label_batch = torch.Tensor(label[index[iteration*self.batch_size:(iteration+1)*self.batch_size]]).cuda()
                pred_batch = self.model(data_batch)
                loss = self.loss(pred_batch.squeeze(1), label_batch)
                loss.backward()
                self.optimizer.step()
                total_num += self.batch_size
                total_loss += loss.item() * self.batch_size
                bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(e, self.epochs, total_loss / total_num))
            self.scheduler.step()
            log_file = open(self.log_path, "a")
            log_file.writelines("Epoch {:4d}/{:4d} | Train Loss: {}\n".format(e, self.epochs, total_loss / total_num))
            log_file.close()
        return self.predict(data)

    def predict(self, pred_data):
        self.model.eval()
        
        pred_results = []
        for index in range(len(pred_data)):
            data_batch = torch.Tensor(pred_data[index:index+1]).cuda()
            pred_batch = self.model(data_batch)
            pred_results.append(pred_batch.detach().cpu().numpy())
        pred_results = np.asarray(pred_results).flatten()
        # print(pred_results.shape)
        return pred_results

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

