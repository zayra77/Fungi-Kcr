#10-fold cross-validation
import matplotlib.pyplot as plt
import numpy as np

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.

def getAUC(file):
    auclist=[]
    f=open(file,encoding='utf-8')
    for line in f:
        if line.startswith('10fold'):
            p=line.rindex(' ')
            auclist.append(round(eval(line[p+1:].strip()),3))
    f.close()
    return auclist
           
Btpr=np.load('result/B_cv_tpr.npy')
Bauc=getAUC('result/B_cv_result')
Ctpr=np.load('result/C_cv_tpr.npy')
Cauc=getAUC('result/C_cv_result')
Ttpr=np.load('result/T_cv_tpr.npy')
Tauc=getAUC('result/T_cv_result')
tpr=np.load('result/cv_tpr.npy')
auc=getAUC('result/cv_result')

plt.figure(figsize=(6,5))
plt.plot(base_fpr,np.average(Btpr,axis=0),label="A (AUC={:.3f})".format(Bauc[0]),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(base_fpr,np.average(Ctpr,axis=0),label="C (AUC={:.3f})".format(Cauc[0]),lw=1,alpha=0.8,linestyle='-',color='orange')
plt.plot(base_fpr,np.average(Ttpr,axis=0),label="T (AUC={:.3f})".format(Tauc[0]),lw=1,alpha=0.8,linestyle='-',color='purple')
plt.plot(base_fpr,np.average(tpr,axis=0),label="All specise (AUC={:.3f})".format(auc[0]),lw=1,alpha=0.8,linestyle='-',color='red')
plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='k')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc=4,fontsize=9)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('cv_roc.jpg',dpi=1000)

plt.show()

######################################
#independent test
import matplotlib.pyplot as plt
import numpy as np

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.

def getAUC(file):
    auclist=[]
    f=open(file,encoding='utf-8')
    for line in f:
        if line.startswith('ind test'):
            p=line.rindex(' ')
            auclist.append(round(eval(line[p+1:].strip()),3))
    f.close()
    return auclist
           
Btpr=np.load('result/B_ind_tpr.npy')
Bauc=getAUC('result/B_ind_result')
Ctpr=np.load('result/C_ind_tpr.npy')
Cauc=getAUC('result/C_ind_result')
Ttpr=np.load('result/T_ind_tpr.npy')
Tauc=getAUC('result/T_ind_result')
tpr=np.load('result/ind_tpr.npy')
auc=getAUC('result/ind_result')

plt.figure(figsize=(6,5))
plt.plot(base_fpr,np.average(Btpr,axis=0),label="A (AUC={:.3f})".format(Bauc[0]),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(base_fpr,np.average(Ctpr,axis=0),label="C (AUC={:.3f})".format(Cauc[0]),lw=1,alpha=0.8,linestyle='-',color='orange')
plt.plot(base_fpr,np.average(Ttpr,axis=0),label="T (AUC={:.3f})".format(Tauc[0]),lw=1,alpha=0.8,linestyle='-',color='purple')
plt.plot(base_fpr,np.average(tpr,axis=0),label="All specise (AUC={:.3f})".format(auc[0]),lw=1,alpha=0.8,linestyle='-',color='red')
plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='k')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc=4,fontsize=9)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('ind_roc.jpg',dpi=1000)

plt.show()

######################################
#10-fold cross-validation in comparison with machine learning
import matplotlib.pyplot as plt
import numpy as np

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.
           
AdaBoosttpr=np.load('ML/Ada_BE_train_mean_tpr.npy')
AdaBoostauc=0.800
LightGBMtpr=np.load('ML/LightGBM_BE_train_mean_tpr.npy')
LightGBMauc=0.830
RFtpr=np.load('ML/RF_BE_train_mean_tpr.npy')
RFauc=0.791
tpr=np.load('result/cv_tpr.npy')
auc=0.904

plt.figure(figsize=(6,5))
plt.plot(base_fpr,AdaBoosttpr,label="AdaBoost (AUC={:.3f})".format(AdaBoostauc),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(base_fpr,LightGBMtpr,label="LightGBM (AUC={:.3f})".format(LightGBMauc),lw=1,alpha=0.8,linestyle='-',color='orange')
plt.plot(base_fpr,RFtpr,label="RF (AUC={:.3f})".format(RFauc),lw=1,alpha=0.8,linestyle='-',color='purple')
plt.plot(base_fpr,np.average(tpr,axis=0),label="our method (AUC={:.3f})".format(auc),lw=1,alpha=0.8,linestyle='-',color='red')
plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='k')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc=4,fontsize=9)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('ML_cv_roc.jpg',dpi=1000)

plt.show()

######################################
#independent test in comparison with machine learning
import matplotlib.pyplot as plt
import numpy as np

base_fpr = np.linspace(0, 1, 101)
base_fpr[-1] = 1.
           
AdaBoosttpr=np.load('ML/Ada_BE_test_mean_tpr.npy')
AdaBoostauc=0.800
LightGBMtpr=np.load('ML/LightGBM_BE_test_mean_tpr.npy')
LightGBMauc=0.837
RFtpr=np.load('ML/RF_BE_test_mean_tpr.npy')
RFauc=0.788
tpr=np.load('result/ind_tpr.npy')
auc=0.901

plt.figure(figsize=(6,5))
plt.plot(base_fpr,AdaBoosttpr,label="AdaBoost (AUC={:.3f})".format(AdaBoostauc),lw=1,alpha=0.8,linestyle='-',color='b')
plt.plot(base_fpr,LightGBMtpr,label="LightGBM (AUC={:.3f})".format(LightGBMauc),lw=1,alpha=0.8,linestyle='-',color='orange')
plt.plot(base_fpr,RFtpr,label="RF (AUC={:.3f})".format(RFauc),lw=1,alpha=0.8,linestyle='-',color='purple')
plt.plot(base_fpr,np.average(tpr,axis=0),label="our method (AUC={:.3f})".format(auc),lw=1,alpha=0.8,linestyle='-',color='red')
plt.plot([0,1],[0,1],lw=1,alpha=0.8,linestyle='--',color='k')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.legend(loc=4,fontsize=9)
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')

plt.savefig('ML_ind_roc.jpg',dpi=1000)

plt.show()