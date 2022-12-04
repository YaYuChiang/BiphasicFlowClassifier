import time, os

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.metrics import f1_score
from scipy.spatial import distance
import pandas as pd

from lib.loglo import init_log
from methods import LR, SVMC, XGB, DT, RF, GB, MLP 
from utils import load_exp_csv, normalize
import values

lo = init_log(__file__, save_log=True, level=os.environ.get("LOG_LEVEL", "0"))

PATH_MODEL = "record/"

class Timer:
    def __init__(self, wedget_lcd) -> None:
        
        self.timer = QTimer()
        self.sec = 0

        self.wedget_lcd = wedget_lcd

        self.timer.timeout.connect(self.LCDEvent)

    def LCDEvent(self):
        self.sec += 1
        self.wedget_lcd.display(self.sec)

class Button:
    def btn_pressed(param):
        print("btn_pressed. ", param)
    
    # should use thread
    def wait_seconds(sec=5):
        print(f"sleep {sec} seconds...")
        time.sleep(sec)
        print("wake up!")

def evaluate(labels, pred):
    cmatrix = confusion_matrix(labels, pred, labels=[0,1])
    tn, fp, fn, tp = cmatrix.ravel()
    acc = (tp+tn) / (tn+fp+fn+tp)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    f1 = f1_score(labels, pred, labels=[0,1])
    return acc, sensitivity, specificity, f1, cmatrix

class WorkerThread(QThread):
    # trigger = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, exp_num, path_csv_train, path_csv_test):
        super().__init__()
        self.exp_num = exp_num
        self.path_csv_train = path_csv_train
        self.path_csv_test = path_csv_test
        
    def run(self):
        # for i in range(5):
        #     time.sleep(1)
        #     self.trigger.emit(str(i+1))
        #     print('WorkerThread::run ' + str(i))
        res_code = Functions.training(self.exp_num, self.path_csv_train, self.path_csv_test)
        self.finished.emit(res_code)

class Functions:
    @staticmethod
    def execute(exp_num : int, method : str, arr_custom_data : list):
        # try:
        #     # str '1, 200, 4, 0.13198, 0.0495, 6.2e-10, 0.245, 7' => float 1.f, 200.f ...
        #     data = [float(x.strip()) for x in arr_custom_data.split(',')]
        # except Exception as ex:
        #     lo.error(f"data format is wrong, {ex}")
        #     return
        
        if exp_num == 1 and len(arr_custom_data) == 8:
            arr_custom_data = arr_custom_data[:-1]
        data = arr_custom_data.copy()
        
        # if exp_num == 1:
        #     if len(data) != 7:
        #         messagebox.showwarning(title="Invalid Input", message="input 7 elements e.g. \"0.0457473, 2.4, 0.0346, 4.34e-10, 200, 0.245, 7\"")
        #         return
        # elif exp_num == 2:
        #     if len(data) != 8:
        #         messagebox.showwarning(title="Invalid Input", message="input 8 elements e.g. \"1, 200, 4, 0.13198, 0.0495, 6.2e-10, 0.245, 7\"")
        #         return
        
        data = np.asarray(data)
        data = np.expand_dims(data, axis=0)
        test_data = normalize(data, exp_num)

        if method == 'LR':
            model = LR(exp_num)
        elif method == 'SVC':
            model = SVMC(exp_num)
        elif method == 'XGB':
            model = XGB(exp_num)
        elif method == 'DT':
            model = DT(exp_num)
        elif method == 'RF':
            model = RF(exp_num)
        elif method == 'GB':
            model = GB(exp_num)
        elif method == 'MLP':
            model = MLP(test_data.shape[1])
        
        if method == 'MLP':
            model.load(os.path.join(PATH_MODEL, f"MLP_{exp_num}.pth"))
            test_result = model.predict(test_data)
        elif method == 'ENS':
            each_test_result = []
            for m in ['XGB', 'DT', 'LR']:
                model = eval(m)(exp_num)
                model.load(os.path.join(PATH_MODEL, f"{m}_{exp_num}"))
                one_test_result = model.predict(test_data)
                each_test_result.append(one_test_result)
            each_test_result = np.asarray(each_test_result)
            test_result = []
            for i in range(each_test_result.shape[1]):
                avg = np.average(each_test_result[:, i])
                if avg >= 0.5:
                    ens_result = 1
                else:
                    ens_result = 0
                test_result.append(ens_result)
            test_result = np.asarray(test_result)
        else:
            model.load(os.path.join(PATH_MODEL, f"{method}_{exp_num}"))
            test_result = model.predict(test_data)
        
        res = None
        if exp_num == 1:
            if method == 'MLP':
                test_result = (test_result*5000).astype(np.int32)
            else:   
                test_result = (test_result).astype(np.int32)
            str_data = ", ".join(map(str, arr_custom_data))
            res = f"exp.{exp_num} method={method} custom_data [{str_data}], res: {test_result[0]}"
            
        elif exp_num == 2:
            test_result = (test_result).astype(bool)
            str_data = ", ".join(map(str, arr_custom_data))
            # res = f"The predicted result is: {test_result[0]}"
            res = f"exp.{exp_num} method={method} custom_data [{str_data}], res: {test_result[0]}"
        
        return res
    
    @staticmethod
    def training(exp_num : int, path_csv_train : str, path_csv_test : str):
        train_data, train_label, test_data, test_label = load_exp_csv(exp_num, path_csv_train, path_csv_test)
        if train_data is None:
            lo.warn("training failed")
            return 1
        
        save_path = "./record/"

        methods = values.dict_methods.get(exp_num, [])
        for method in methods:
            if method == 'LR':
                model = LR(exp_num)
            elif method == 'SVC':
                model = SVMC(exp_num)
            elif method == 'XGB':
                model = XGB(exp_num)
            elif method == 'DT':
                model = DT(exp_num)
            elif method == 'RF':
                model = RF(exp_num)
            elif method == 'GB':
                model = GB(exp_num)
            elif method == 'MLP':
                model = MLP(train_data.shape[1], epochs=150, batch_size=16)

            if method == 'MLP':
                train_result = model.fit(train_data, train_label/5000.0)
                model.save(os.path.join(save_path, f"MLP_{exp_num}-1.pth"))
                model.load(os.path.join(save_path, f"MLP_{exp_num}-1.pth"))
                test_result = model.predict(test_data)
            else:
                train_result = model.fit(train_data, train_label)
                model.save(os.path.join(save_path, f"{method}_{exp_num}"))
                model.load(os.path.join(save_path, f"{method}_{exp_num}"))
                test_result = model.predict(test_data)
            
            if exp_num == 1:
                if method == 'MLP':
                    # train_result *= 5000
                    # train_label *= 5000
                    test_result = (test_result*5000).astype(np.int32)
                else:
                    test_result = (test_result).astype(np.int32)
                test_label = (test_label).astype(np.int32)
                print(test_result)
                df = pd.DataFrame({'label': test_label, 'pred': test_result})
                df.to_csv("./result/exp_{}_{}.csv".format(exp_num, method), index=False)
                mse = mean_squared_error(train_label, train_result)
                mdistance = distance.minkowski(train_label, train_result)
                print("Train MSE", mse)
                print("Train mDist.", mdistance)
                mse = mean_squared_error(test_label, test_result)
                mdistance = distance.minkowski(test_label, test_result)
                print("Test MSE", mse)
                print("Test mDist.", mdistance)
                
            elif exp_num == 2:
                test_result = (test_result).astype(np.int32)
                print(test_result)
                df = pd.DataFrame({'label': test_label, 'pred': test_result})
                df.to_csv("./result/exp_{}_{}.csv".format(exp_num, method), index=False)
                acc, sensitivity, specificity, f1, cmatrix = evaluate(train_label, train_result)
                print("Train Acc", acc, "Sensitivity", sensitivity, "Specificity", specificity, "F1", f1)
                print(cmatrix)
                acc, sensitivity, specificity, f1, cmatrix = evaluate(test_label, test_result)
                print("Test Acc", acc, "Sensitivity", sensitivity, "Specificity", specificity, "F1", f1)
                print(cmatrix)
        return 0