import sys 
sys.path.append('/home/aistudio/external-libraries')
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import numpy as np
#import pickle
import math
import random
from sklearn import metrics
from numpy import interp
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'

"""
十次独立测试
"""


train_filepath= sys.argv[1]
test_filepath= sys.argv[2]
#dict_filepath= '/home/aistudio/work/khib/Datasets/residue2idx.pkl'

'''
def get_word2id_dict(dict_filepath):
    residue2idx = pickle.load(open(dict_filepath ,'rb'))
    word2id_dict=residue2idx
    return word2id_dict
word2id_dict=get_word2id_dict(dict_filepath)
'''
word2id_dict=dict(zip(Amino_acid_sequence,range(21)))

def encode(file_path):
    codes = []
    labels = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            name,seq, y_label = line.split()
            code = [word2id_dict.get(AA) for AA in seq]
            label = int(y_label)
            codes.append(code)
            labels.append(label)
    return np.array(codes).astype('int32'), labels

train_feature,train_label=encode(train_filepath)
test_feature,test_label=encode(test_filepath)

# word to id
class MyDataset(Dataset):

    def __init__(self, feature, label):
        super(MyDataset, self).__init__()
        self.feature = feature
        self.label=label

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]
    def __len__(self):
        return len(self.feature)

#原编码加上位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_size, dropout=0.2, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_size)
        pos_mat=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(dim=1)
        i_mat=torch.pow(10000,torch.arange(0,embedding_size,2,dtype=torch.float32)/ embedding_size)
        pe[:,0::2] = torch.sin(pos_mat / i_mat)
        pe[:,1::2] = torch.cos(pos_mat / i_mat)
        pe=pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x=x+self.pe[:x.size(0),:]
        return self.dropout(x)
    
#自注意力，单头
class Self_Attention(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(Self_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        query = self.query(x).view(batch_size, seq_len, self.hidden_size)
        key = self.key(x).view(batch_size, seq_len, self.hidden_size)
        value = self.value(x).view(batch_size, seq_len, self.hidden_size)
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)
        atten_scores = torch.bmm(query, key.transpose(1, 2)) #===>[bacth_size,seq_len,seq_len]
        atten_scores = torch.softmax(atten_scores, dim=-1)
        context = torch.bmm(atten_scores, value)
        return context

#WE+PE+CNN+self-attention+GRU
class CNN_BiGRU_Model(nn.Module):
    #num_heads == d_model
    def __init__(self,vocab_size,embedding_dim, num_classes=2,**kwargs):
        super(CNN_BiGRU_Model,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_dim
        self.token_embedding=nn.Embedding(self.vocab_size,self.embedding_size)
        self.positional_embedding=PositionalEncoding(self.embedding_size)
        self.encode=nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_size, out_channels=128, kernel_size=7,stride=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
        )
        self.attention=Self_Attention(hidden_size=64,dropout_rate=0.2)
        self.BiGRU_layer = nn.GRU(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout=nn.Dropout(0.5)
        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(19 * 64,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,num_classes),
        )
    def forward(self,x,valid_lens=None):
        token_out = self.token_embedding(x)
        adapt_x = self.positional_embedding(token_out.transpose(0, 1)).transpose(0, 1)  # (batch_size,seq_len,d_model)
        adapt_x=adapt_x.transpose(1,2)
        conv_out=self.encode(adapt_x)
        conv_out=conv_out.transpose(1,2)
        BiGRU_out, c = self.BiGRU_layer(conv_out)
        BiGRU_out = self.dropout(BiGRU_out)
        #att_out=self.attention(BiGRU_out)
        out = self.fc(BiGRU_out)
        return x,out

#依预测值和真实值，计算TN，FP，FN,TP,sn,sp,acc,mcc,pr,recall,f1score
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
    Pr = TP / (TP + FP)
    recall = metrics.recall_score(y_test_true, y_pred_label)
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    return (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)


# mean and std of metrics values:
def Calcuate_mean_std_metrics_values(total_SN,total_SP,total_ACC,total_F1_score,total_Pr,total_MCC,total_AUC):
    # Calculate mean and std metrics values:
    mean_SN = np.mean(total_SN)
    mean_SP = np.mean(total_SP)
    mean_ACC = np.mean(total_ACC)
    mean_F1_score = np.mean(total_F1_score)
    mean_Pr = np.mean(total_Pr)
    mean_MCC = np.mean(total_MCC)
    mean_AUC = np.mean(total_AUC)
    std_SN = np.std(total_SN)
    std_SP = np.std(total_SP)
    std_ACC = np.std(total_ACC)
    std_F1_score = np.std(total_F1_score)
    std_Pr = np.std(total_Pr)
    std_MCC = np.std(total_MCC)
    std_AUC = np.std(total_AUC)
    mean_metrics = []
    mean_metrics.append(mean_SN)
    mean_metrics.append(mean_SP)
    mean_metrics.append(mean_ACC)
    mean_metrics.append(mean_F1_score)
    mean_metrics.append(mean_Pr)
    mean_metrics.append(mean_MCC)
    mean_metrics.append(mean_AUC)
    std_metrics = []
    std_metrics.append(std_SN)
    std_metrics.append(std_SP)
    std_metrics.append(std_ACC)
    std_metrics.append(std_F1_score)
    std_metrics.append(std_Pr)
    std_metrics.append(std_MCC)
    std_metrics.append(std_AUC)
    #np.save('../np_weights/DeepFungiKhib_ind_test_mean_Metric.npy', mean_metrics)
    #np.save('../np_weights/DeepFungiKhib_ind_test_std_Metric.npy', std_metrics)
    print(
        "ind test Mean metric : SN is {:.3f},SP is {:.3f},Pr is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f}".
        format(mean_SN, mean_SP, mean_Pr,mean_ACC, mean_F1_score, mean_MCC, mean_AUC))
    print(
        "ind test std metric : SN is {:.4f},SP is {:.4f},Pr is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f}".
        format(std_SN, std_SP, std_Pr,std_ACC, std_F1_score,std_MCC, std_AUC))

def train(model,epochs,train_loader,test_loader, optimizer, criterion,device):
    train_loss_min=1.0
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        model.train()
        for batch_id, data in enumerate(train_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            _,y_predict = model(x_data)
            loss = criterion(y_predict,y_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            x_data = torch.tensor(train_feature).to(device)
            y_data = torch.tensor(train_label, dtype=torch.long).to(device)
            _,y_predict = model(x_data)
            train_loss = criterion(y_predict,y_data)
            print('training loss is {}'.format(train_loss))
            if train_loss < train_loss_min:
                train_loss_min = train_loss
                torch.save(model.state_dict(),'DFmodel.pth')
    x_data = torch.tensor(test_feature).to(device)
    model.load_state_dict(torch.load('DFmodel.pth', map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
    model.eval()
    _,y_predict = model(x_data)
                # the max possible label
    y_predict_label = torch.argmax(y_predict, dim=1)
    y_predict_label = y_predict_label.detach().cpu().numpy()
    y_test_true = np.array(test_label)
    y_test_score = y_predict[:, 1].detach().cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(y_test_true, y_test_score)
    res_auc = metrics.auc(fpr, tpr)
    #np.save('../np_weights/DeepFungiKhib_y_test_true(test).npy', y_test_true_list)
    #np.save('../np_weights/DeepFungiKhib_y_test_score(test).npy', y_test_score_list)
    #np.save('../np_weights/DeepFungiKhib_y_test_pred(test).npy', y_test_pred_label_list)
    #confusion matrix
    (TN,TP,FN,FP),(SN,SP,ACC,MCC,Pr,recall,F1Score)=Calculate_confusion_matrix(y_test_true,y_predict_label)
    print("[Valid TP is {},FP is {},TN is {},FN is {}]".format(TP, FP, TN, FN))
    print("Valid : SN is {},SP is {},Pr is {},ACC is {},F1-score is {},MCC is {},\n AUC is {}, recall is {}".
        format(SN, SP, Pr, ACC, F1Score, MCC, res_auc, recall))
            
    test_roc.append([fpr, tpr])
    test_roc_auc_areas.append(res_auc)
    #mean tpr
    tpr = interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    total_SN.append(SN)
    total_SP.append(SP)
    total_ACC.append(ACC)
    total_F1_score.append(F1Score)
    total_Pr.append(Pr)
    total_MCC.append(MCC)
    total_AUC.append(res_auc)

def random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

if __name__ == '__main__':

    # total_metrics
    total_SN = []
    total_SP = []
    total_ACC = []
    total_F1_score = []
    total_Pr=[]
    total_MCC = []
    total_AUC = []

    # figure mean AUC:
    test_roc = []
    test_roc_auc_areas = []
    mean_fpr = np.linspace(0, 1, 101)
    mean_fpr[-1] = 1.0
    tprs = []

    vocab_size = len(word2id_dict)
    embedding_dim =128
    num_heads = 8
    d_model=256
    d_k = 256 # 32
    d_v = 256 # 32
    # the number of epochs is 50
    epochs =100
    batch_size = 128
    learn_rate = 0.001

    # paramters :
    num_classes = 2
    num_layers = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Dataset:
    train_set = MyDataset(train_feature, train_label)
    test_set = MyDataset(test_feature, test_label)
    # DataLoader:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
    seed = 800
    for i in range(10):
        print("第{}次独立测试".format(i + 1))
        random_seeds(seed)
        seed += 10
        model = CNN_BiGRU_Model(vocab_size=vocab_size,embedding_dim=embedding_dim,num_layers=num_layers,dropout=0.5,num_classes=num_classes)
        model.to(device)
        #print(model)
        #optimizer
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate,weight_decay=5e-4)
        #focal loss
        # train_criterion = FocalLoss(alpha=0.65,gamma=1)
        criterion = nn.CrossEntropyLoss()
        #training:
        train(model,epochs,train_loader,test_loader, optimizer,criterion,device)
        # load model to test
        '''model_path = '../DL_weights/CNN-BiGRU-Khib.pth'
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        indepedent_test(model,test_loader,criterion,device)'''

    Calcuate_mean_std_metrics_values(total_SN, total_SP, total_ACC, total_F1_score,total_Pr, total_MCC, total_AUC)
    np.save(sys.argv[3],np.array(tprs))
    #Kf_AUROC_show(mean_fpr, tprs, test_roc_auc_areas)
