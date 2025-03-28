from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,auc,roc_auc_score,roc_curve
import numpy as np
import math
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from numpy import interp
import random
import warnings
warnings.filterwarnings("ignore")

AA_Seq='ACDEFGHIKLMNPQRSTVWYX'

train_path="../data/train"
ind_test_path="../data/test"

def read_file(filepath):

    data=[]
    with open(filepath,mode='r',encoding='utf-8') as f:

        for line in f.readlines():
            _,seq,label=line.strip().split()
            data.append((seq,label))
        f.close()
    return data

def get_Binary_encoding(data):

    X=[]
    y=[]
    for seq,label in data:
        one_code=[]
        for i in seq:
            vector=[0]*21
            vector[AA_Seq.index(i)]=1

            one_code.append(vector)
        X.append(one_code)
        y.append(int(label))

    X=np.array(X)
    n,seq_len,dim=X.shape
    # reshape
    X=np.reshape(X,(n,seq_len * dim))
    print("new X shape :",X.shape)
    y=np.array(y)
    print(y.shape)
    return X,y

def Calculate_confusion_matrix(y_test_true,y_pred_label):

    conf_matrix = confusion_matrix(y_test_true, y_pred_label)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1Score = (2 * TP) / float(2 * TP + FP + FN)

    return (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score)

def Calcuate_mean_std_metrics_values(total_SN,total_SP,total_ACC,total_F1_score,total_MCC,total_AUC):
    # Calculate mean and std metrics values:
    mean_SN = np.mean(total_SN)
    mean_SP = np.mean(total_SP)
    mean_ACC = np.mean(total_ACC)
    mean_F1_score = np.mean(total_F1_score)
    mean_MCC = np.mean(total_MCC)
    mean_AUC = np.mean(total_AUC)

    std_SN = np.std(total_SN)
    std_SP = np.std(total_SP)
    std_ACC = np.std(total_ACC)
    std_F1_score = np.std(total_F1_score)
    std_MCC = np.std(total_MCC)
    std_AUC = np.std(total_AUC)

    mean_metrics = []
    mean_metrics.append(mean_SN)
    mean_metrics.append(mean_SP)
    mean_metrics.append(mean_ACC)
    mean_metrics.append(mean_F1_score)
    mean_metrics.append(mean_MCC)
    mean_metrics.append(mean_AUC)

    std_metrics = []
    std_metrics.append(std_SN)
    std_metrics.append(std_SP)
    std_metrics.append(std_ACC)
    std_metrics.append(std_F1_score)
    std_metrics.append(std_MCC)
    std_metrics.append(std_AUC)

    print(
        "ind test Mean metrics : SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".
        format(mean_SN, mean_SP, mean_ACC, mean_F1_score, mean_MCC, mean_AUC))
    print(
        "ind test std metrics : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".
        format(std_SN, std_SP, std_ACC, std_F1_score, std_MCC, std_AUC))

def cross_validation(train_data):
    X_train,y_train=train_data
    train_SN=[]
    train_SP=[]
    train_ACC=[]
    train_F1_score=[]
    train_MCC=[]
    train_AUC=[]
    kf = KFold(n_splits=10, shuffle=True)
    fold = 1
    train_tprs=[]
    for train_index, valid_index in kf.split(X_train, y_train):
        print("第{}次交叉验证开始...".format(fold))
        this_train_x, this_train_y = X_train[train_index], y_train[train_index]
        this_valid_x, this_valid_y = X_train[valid_index], y_train[valid_index]
        ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada_model.fit(this_train_x, this_train_y)
        # train fit
        y_train_pred = ada_model.predict(this_train_x)
        # valid fit
        y_valid_pred = ada_model.predict(this_valid_x)
        # acc:
        train_acc = accuracy_score(this_train_y, y_train_pred)
        valid_acc = accuracy_score(this_valid_y, y_valid_pred)
        #print("训练集准确率: {:.2f}%".format(train_acc * 100))
        #print("验证集准确率: {:.2f}%".format(valid_acc * 100))
        # auc:


        # 5Kfold SN、SP、ACC、F1_score,MCC
        y_valid_true_label = this_valid_y
        y_valid_score=ada_model.predict_proba(this_valid_x)[:, 1]
        y_valid_pred_label = y_valid_pred

        print("混淆矩阵")
        (TN, TP, FN, FP), (SN, SP, ACC, MCC, F1Score) = Calculate_confusion_matrix(y_valid_true_label,
                                                                                   y_valid_pred_label)
        valid_auc=roc_auc_score(y_valid_true_label,y_valid_score)
        fpr, tpr, _ = roc_curve(y_valid_true_label,y_valid_score)
        tpr = interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        train_tprs.append(tpr)

        train_SN.append(SN)
        train_SP.append(SP)
        train_ACC.append(ACC)
        train_F1_score.append(F1Score)
        train_MCC.append(MCC)
        train_AUC.append(valid_auc)

        print("Train TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Train SN is {},SP is {},ACC is {},F1-score is {}, MCC is {},AUC is {}".format(SN, SP, ACC, F1Score, MCC,valid_auc))

        fold += 1

    print(
        "Train Mean metrics values: SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".format(np.mean(train_SN),
                                                                                   np.mean(train_SP),
                                                                                   np.mean(train_ACC),
                                                                                   np.mean(train_F1_score),
                                                                                   np.mean(train_MCC),np.mean(train_AUC)))
    print("Train Std metrics values : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".format(np.std(train_SN),
                                                                                    np.std(train_SP),
                                                                                    np.std(train_ACC),
                                                                                    np.std(train_F1_score),
                                                                                    np.std(train_MCC),
                                                                                    np.std(train_AUC)))
    mean_tpr=np.mean(train_tprs,axis=0)
    np.save('Ada_BE_train_mean_tpr.npy', mean_tpr)

def independent_test(ada_model,X_train,y_train,X_test,y_test,random_seed):
    #print("-----------------------------ind_test start------------------------")

    X_train,y_train=shuffle(X_train,y_train,random_state=random_seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_seed)
    #fit
    ada_model.fit(X_train, y_train)
    #predict
    y_test_pred_label= ada_model.predict(X_test)
    # ind test auc:
    y_test_score=ada_model.predict_proba(X_test)[:, 1]
    # calculate SN、SP、ACC、MCC
    y_test_true_label = y_test
    y_test_pred_label = y_test_pred_label
    fpr, tpr, _ = roc_curve(y_test, y_test_score)
    test_auc = metrics.auc(fpr, tpr) #y_test_true, y_test_score
    print("test auc :", test_auc)
    # mean tpr and fpr
    tpr = interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

    (TN, TP, FN, FP), (SN, SP, ACC, MCC, F1Score) = Calculate_confusion_matrix(y_test_true_label, y_test_pred_label)
    total_SN.append(SN)
    total_SP.append(SP)
    total_ACC.append(ACC)
    total_F1_score.append(F1Score)
    total_MCC.append(MCC)
    total_AUC.append(test_auc)

    print("ind test: TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
    print("ind test: SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".format(SN, SP, ACC, F1Score, MCC,test_auc))

def ind_test_10time(train_data,ind_test_data):

    X_train,y_train=train_data
    X_test,y_test=ind_test_data

    # ind test:
    random_seed=42
    for i in range(10):
        np.random.seed(random_seed)
        ada_clf = AdaBoostClassifier(n_estimators=100,random_state=random_seed)
        random_seed += 10
        independent_test(ada_clf,X_train,y_train,X_test,y_test,random_seed)
    Calcuate_mean_std_metrics_values(total_SN, total_SP, total_ACC, total_F1_score, total_MCC, total_AUC)
    #save mean tprs and fprs
    mean_tpr=np.mean(tprs,axis=0)
    np.save('Ada_BE_test_mean_tpr.npy', mean_tpr)



train=read_file(train_path)
ind_test=read_file(ind_test_path)

#binary encode:
train_data=get_Binary_encoding(train)
ind_test_data=get_Binary_encoding(ind_test)

# cross_validation
mean_fpr = np.linspace(0, 1, 101)
mean_fpr[-1] = 1.0
cross_validation(train_data)

#independent test
total_SN = []
total_SP = []
total_ACC = []
total_F1_score = []
total_MCC = []
total_AUC = []
tprs=[]
ind_test_10time(train_data,ind_test_data)
