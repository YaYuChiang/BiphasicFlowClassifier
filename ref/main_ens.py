import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, mean_squared_error
from methods import *
from utils import *
from scipy.spatial import distance
from sklearn.metrics import f1_score

def evaluate(labels, pred):
    cmatrix = confusion_matrix(labels, pred, labels=[0,1])
    tn, fp, fn, tp = cmatrix.ravel()
    acc = (tp+tn) / (tn+fp+fn+tp)
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    f1 = f1_score(labels, pred, labels=[0,1])
    return acc, sensitivity, specificity, f1, cmatrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_num", '-e', type=int, default=2, help="experiment number [1,2]")
    parser.add_argument("--method", '-m', nargs='+', type=str, default=None, help="Classification Method [LR, SVC, XGB, DT, MLP, GB, RF]")

    args = parser.parse_args()
    # assert args.method in ['LR', 'SVC', 'XGB', 'DT', 'MLP'], 'undefined classification method'

    print("========= Run Experiment ============")
    print("===== exp {} with {} =============".format(args.exp_num, args.method))
    if args.exp_num == 1:
        train_data, train_label, test_data, test_label = load_exp1_data()
    elif args.exp_num == 2:
        train_data, train_label, test_data, test_label = load_exp2_data()
    # print(train_data.shape)
    # print(train_label.shape)
    # print(test_data.shape)
    # print(test_label.shape)


    # train_index, test_index = train_test_split(range(len(data)), test_size=0.15)
    # train_index, val_index = train_test_split(train_index, test_size=0.15)
    # train_data = data[train_index]
    # train_label = label[train_index]
    # test_data = data[test_index]
    # test_label = label[test_index]
    each_train_result = []
    each_test_result = []

    for m in args.method:
        if m == 'LR':
            model = LR(args.exp_num)
        elif m == 'SVC':
            model = SVMC(args.exp_num)
        elif m == 'XGB':
            model = XGB(args.exp_num)
        elif m == 'DT':
            model = DT(args.exp_num)
        elif m == 'RF':
            model = RF(args.exp_num)
        elif m == 'GB':
            model = GB(args.exp_num)
        elif m == 'MLP':
            model = MLP(train_data.shape[1], epochs=150, batch_size=16)

        if m == 'MLP':
            train_result = model.fit(train_data, train_label/5000.0)
            test_result = model.predict(test_data)
        else:
            train_result = model.fit(train_data, train_label)
            test_result = model.predict(test_data)
        
        each_train_result.append(train_result)
        each_test_result.append(test_result)
    each_train_result = np.asarray(each_train_result)
    each_test_result = np.asarray(each_test_result)
    
    final_train_result = []
    for i in range(each_train_result.shape[1]):
        avg = np.average(each_train_result[:, i])
        if avg >= 0.5:
            ens_result = 1
        else:
            ens_result = 0
        # print(each_train_result[:, i], ens_result)
        final_train_result.append(ens_result)

    final_test_result = []
    for i in range(each_test_result.shape[1]):
        avg = np.average(each_test_result[:, i])
        if avg >= 0.5:
            ens_result = 1
        else:
            ens_result = 0
        # print(each_test_result[:, i], ens_result)
        final_test_result.append(ens_result)

    final_train_result = np.asarray(final_train_result)
    final_test_result = np.asarray(final_test_result)

    if args.exp_num == 1:
        if args.method == 'MLP':
            # train_result *= 5000
            # train_label *= 5000
            test_result = (test_result*5000).astype(np.int32)
        else:
            test_result = (test_result).astype(np.int32)
        test_label = (test_label).astype(np.int32)
        print(test_result)
        df = pd.DataFrame({'label': test_label, 'pred': test_result})
        df.to_csv("../result/exp_{}_{}.csv".format(args.exp_num, args.method), index=False)
        mse = mean_squared_error(train_label, train_result)
        mdistance = distance.minkowski(train_label, train_result)
        print("Train MSE", mse)
        print("Train mDist.", mdistance)
        mse = mean_squared_error(test_label, test_result)
        mdistance = distance.minkowski(test_label, test_result)
        print("Test MSE", mse)
        print("Test mDist.", mdistance)
        
    elif args.exp_num == 2:
        final_test_result = (final_test_result).astype(np.int32)
        print(final_test_result)
        df = pd.DataFrame({'label': test_label, 'pred': final_test_result})
        df.to_csv("../result/exp_{}_{}.csv".format(args.exp_num, args.method), index=False)
        acc, sensitivity, specificity, f1, cmatrix = evaluate(train_label, train_result)
        print("Train Acc", acc, "Sensitivity", sensitivity, "Specificity", specificity, "F1", f1)
        print(cmatrix)
        acc, sensitivity, specificity, f1, cmatrix = evaluate(test_label, final_test_result)
        print("Test Acc", acc, "Sensitivity", sensitivity, "Specificity", specificity, "F1", f1)
        print(cmatrix)