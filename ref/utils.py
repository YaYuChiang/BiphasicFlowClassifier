import pandas as pd
import numpy as np

def load_exp1_data():
    df = pd.read_csv("../training_data/exp1.csv")
    df_np = df.to_numpy().astype(np.float32)
    gt = df_np[..., -1]
    data = df_np[..., :-1]

    df = pd.read_csv("../test_data/exp1.csv")
    df_np = df.to_numpy().astype(np.float32)
    test_gt = df_np[..., -1]
    test_data = df_np[..., :-1]
    data = normalize(data, 1)
    test_data = normalize(test_data, 1)
    return data, gt, test_data, test_gt

def load_exp2_data():
    df = pd.read_csv("../training_data/exp2.csv")
    df_np = df.to_numpy().astype(np.float32)
    gt = df_np[..., -1].astype(np.int)
    data = df_np[..., :-4]

    df = pd.read_csv("../test_data/exp2.csv")
    df_np = df.to_numpy().astype(np.float32)
    test_gt = df_np[..., -1].astype(np.int)
    test_data = df_np[..., :-1]
    data = normalize(data, 2)
    test_data = normalize(test_data, 2)
    return data, gt, test_data, test_gt

def normalize(data, exp_num):
    maxvalue_dict={1:[0.25161028, 2.4, 651.17883, 1.2032408, 1100.0, 0.453, 14.0], 2:[3.0, 2000.0, 4.400000095367432, 1.319804072380066, 651.0, 72.0, 0.453000009059906, 14.0]}
    minvalue_dice={1:[0.045156743, 0.0, 0.0346, 4.34e-10, 200.0, 0.245, 7.0], 2:[1.0, 200.0, 0.0, 0.0, 0.02500000037252903, 3.1499999830764125e-10, 0.0, 7.0]}
    for i in range(data.shape[1]):
        # maxvalue = np.amax([np.amax(data[:, i]), np.amax(test_data[:, i])])
        maxvalue = maxvalue_dict[exp_num][i]
        # minvalue = np.amin([np.amin(data[:, i]), np.amin(test_data[:, i])])
        minvalue = minvalue_dice[exp_num][i]
        # print(maxvalue, minvalue)
        data[:, i] = (data[:, i] - minvalue) / (maxvalue - minvalue)
    return data
    