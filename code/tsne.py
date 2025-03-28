import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
 
# 加载示例数据集
y=[]
f=open('../data/test')
for line in f:
    col=line.strip().split()
    y.append(int(col[-1]))
f.close()
y=np.array(y)
X = np.load('binary_encode_test.npy')

 
# 实例化TSNE模型，并将数据降维到2维
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
 
# 用不同颜色表示不同类别
plt.figure(figsize=(5,5))
plt.plot(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
         'bo', ms=2,mfc='white',label='non-Kcr')
plt.plot(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
         'ro', ms=2,mfc='white',label='Kcr')
plt.legend(fontsize=8,loc=4)
plt.savefig('tsne1.jpg',dpi=600)

# 实例化TSNE模型，并将数据降维到2维
X=np.load('fc_pred_test.npy')
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
 
# 用不同颜色表示不同类别
plt.figure(figsize=(5,5))
plt.plot(X_tsne[y == 0, 0], X_tsne[y == 0, 1],
         'bo', ms=2,mfc='white',label='non-Kcr')
plt.plot(X_tsne[y == 1, 0], X_tsne[y == 1, 1],
         'ro', ms=2,mfc='white',label='Kcr')
plt.legend(fontsize=8,loc=4)
plt.savefig('tsne2.jpg',dpi=600)

