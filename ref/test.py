import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, mean_squared_error
from methods import *
from utils import *
from scipy.spatial import distance

def evaluate(labels, pred):
    cmatrix = confusion_matrix(labels, pred, labels=[0,1])
    tn, fp, fn, tp = cmatrix.ravel()
    acc = (tp+tn) / (tn+fp+fn+tp)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    return acc, sensitivity, specificity, cmatrix


if __name__ == '__main__':

    print("===== Please select the experiment =========")
    print("1: Maximum Input Flow Rate Prediction")
    print("2: Prediction of Droplet Separation")
    exp_num = input("Select:")
    exp_num = int(exp_num)
    if exp_num == 1:
        train_data, train_label, test_data, test_label = load_exp1_data()
    elif exp_num == 2:
        train_data, train_label, test_data, test_label = load_exp2_data()
    print("========= Please type in the custom data (seperated by comma) ======")
    if exp_num == 1:
        print("ca,polarity,bo,mo,org.flowrate,pitch,L")
        data = input()
        
    elif exp_num == 2:
        print("ratio,total flow rate,polarity,ca,bo,mo,pitch,l")
        data = input()
    data = [float(d) for d in data.split(",")]
    data = np.asarray(data)
    data = np.expand_dims(data, axis=0)
    test_data = normalize(data, exp_num)

    method_list1 = ['LR', 'SVC', 'XGB', 'DT', 'GB', 'RF', 'MLP']
    method_list2 = ['LR', 'SVC', 'XGB', 'DT', 'GB', 'RF', 'ENS']
    if exp_num == 1:
        print("===== Please select a method =========")
        print("1: LR, 2: SVC, 3: XGB, 4: DT, 5: GB, 6: RF, 7: MLP")
        method_idx = input("Select:")
        method_idx = int(method_idx)
        method = method_list1[method_idx-1]
    elif exp_num == 2:
        print("===== Please select a method =========")
        print("1: LR, 2: SVC, 3: XGB, 4: DT, 5: GB, 6: RF, 7: ENS")
        method_idx = input("Select:")
        method_idx = int(method_idx)
        method = method_list2[method_idx-1]
    
    save_path = "../record/"

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
        model.load(os.path.join(save_path, f"MLP_{exp_num}.pth"))
        test_result = model.predict(test_data)
    elif method == 'ENS':
        each_test_result = []
        for m in ['XGB', 'DT', 'LR']:
            model = eval(m)(exp_num)
            model.load(os.path.join(save_path, f"{m}_{exp_num}"))
            one_test_result = model.predict(test_data)
            each_test_result.append(one_test_result)
        each_test_result = np.asarray(each_test_result)
        test_result = []
        for i in range(each_test_result.shape[1]):
            # print(each_test_result[:, i])
            avg = np.average(each_test_result[:, i])
            if avg >= 0.5:
                ens_result = 1
            else:
                ens_result = 0
            test_result.append(ens_result)
        test_result = np.asarray(test_result)
    else:
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
        print("The predicted result is:", test_result)
        # mse = mean_squared_error(test_label, test_result)
        # mdistance = distance.minkowski(test_label, test_result)
        # print("Test MSE", mse)
        # print("Test mDist.", mdistance)
        
    elif exp_num == 2:
        test_result = (test_result).astype(bool)
        print("The predicted result is:", test_result)
        # acc, sensitivity, specificity, cmatrix = evaluate(test_label, test_result)
        # print("Test Acc", acc, "Sensitivity", sensitivity, "Specificity", specificity)
        # print(cmatrix)
