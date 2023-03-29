from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics, preprocessing
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import psutil
import cProfile


import time

link = "./train.csv"
df = pd.read_csv(link)
X = np.array(df)

X.shape #確認

#ラベル作成
nomal = [1]*200
anomaly = [0]*200

nomal.extend(anomaly)
y_train = np.array(nomal)
y = y_train.astype("uint8")


#特徴量となるデータを3つに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

#モデル及び学習
start_time = time.time()#処理時間を見たい場合は記載
model=svm.LinearSVC(C=15.0,
                    verbose=1,
                    #dual=False,  
                    tol=0.01,
                    random_state=40)
model.fit(X_train,y_train)
end_time = time.time()#処理時間を見たい場合は記載
cProfile.run('model.fit(X, y)')#処理時間/詳細を見たい場合は記載
process = psutil.Process()#処理時間/詳細を見たい場合は記載
print(process.cpu_percent())#処理時間/詳細を見たい場合は記載

#学習にかかった時間を確認できる
elapsed_time = end_time - start_time
print(f'Elapsed time: {elapsed_time:.4f} sec')

#イタレーション数を確認する事ができる。今回は235
n_iter = model.n_iter_
print(f'Number of iterations: {n_iter}')

#訓練データと検証データ学習データの結果を示すF値
y_pred = model.predict(X_val)
f1_score(y_true=y_val, y_pred=y_pred)
#0.958904109589041

print(accuracy_score(y_true=y_val, y_pred=y_pred))
#学習精度0.9625

#混合行列
confusion_matrix(y_pred=y_pred,y_true=y_val)

#array([[42,  0],
#       [ 3, 35]], dtype=int64)

#########テストデータの場合#########

y_pred2 = model.predict(X_test)
f1_score(y_true=y_test, y_pred=y_pred2)
#0.9565217391304348

confusion_matrix(y_pred=y_pred2,y_true=y_test)
#array([[32,  2],
#       [ 2, 44]], dtype=int64)

pd.DataFrame(data={'y_pred':y_pred2,'y_test':y_test})


#ROC曲線を用いたAUC評価を行う
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
roc = roc_curve(y_test,y_pred2)
fpr, tpr, thresholds = roc_curve(y_test,y_pred2)
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()

from sklearn.metrics import auc
auc(fpr, tpr)
#0.948849104859335

from sklearn.metrics import classification_report
print(classification_report(y_pred2,y_test))  






