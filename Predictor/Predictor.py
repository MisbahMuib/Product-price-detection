import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import time
startTime = time.time()

pick_in  = open('../Database/pickle/dataset.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
x = []
y = []
for feature, label in data:
    x.append(feature)
    y.append(label)

#datauserate
print( 'total training image',len(x))
thirtypercent=0.30  # training size 70%
fourtypercent=0.40   # training size 60%
fiftypercent=0.50    # training size 50%
sixtypercent=0.60    # training size 40%
seventypercent=0.70   # training size 30%


################# Validation dataset ###################
pick_in  = open('../Database/pickle/Vdataset.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
random.shuffle(data)
vx = []
vy = []
for feature, label in data:
    vx.append(feature)
    vy.append(label)

#datauserate
print('total testing image',len(vx))

print("\n########## SVM algorithm ###########")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
clf=svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_pred=clf.predict(vx)
print("test size=30, accuracy = {0:.2f}".format(100*metrics.accuracy_score(vy, y_pred)),"%")


harpic, lux, Toothpowder, pepsodent, vim = list(), list(),list(),list(), list()
Pharpic, Plux, PToothpowder, Ppepsodent, Pvim = list(), list(),list(),list(), list()

df = pd.DataFrame({'Real': vy, 'Predicted':y_pred})

for i in vy:
    if i == 0:
        harpic.append(0)
    elif i == 1:
        lux.append(1)
    elif i == 2:
        Toothpowder.append(2)
    elif i == 3:
        pepsodent.append(3)
    elif i == 4:
        vim.append(4)



predharpic = df.loc[(df['Real']==0) & (df['Predicted'] == 0)]
predlux = df.loc[(df['Real'] == 1) & (df['Predicted'] == 1)]
predToothpowder = df.loc[(df['Real'] == 2) & (df['Predicted'] == 2)]
predpepsodent = df.loc[(df['Real'] == 3) & (df['Predicted'] == 3)]
predvim = df.loc[(df['Real'] == 4) & (df['Predicted'] == 4)]


harpicLis = [len(harpic),len(predharpic)]
luxLis = [len(lux),len(predlux)]
ToothpowderLis = [len(Toothpowder),len(predToothpowder)]
pepsodentLis = [len(pepsodent),len(predpepsodent)]
vimLis = [len(vim),len(predvim)]
predictdata = [harpicLis,luxLis,ToothpowderLis,pepsodentLis,vimLis,]


# Creates pandas DataFrame.
predictdf = pd.DataFrame(predictdata,index=['harpic', 'lux','Toothpowder','pepsodent','vim'],columns=['Real','Prediction'])
#it create 3 columns
predictdf.plot.bar(rot=0,) #rot write lebel horizontally
plt.xlabel('Product')
plt.yticks([x for x in range(1,26) if x%2==0])
plt.ylabel('Numbers')
plt.grid()
plt.savefig('Real price and predicted.png') # need to call before calling show
plt.show()


############################# Confusion Matrix #########################
#
#
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# plt.rcParams["font.family"] = "Times New Roman"
# # plt.rcParams["figure.figsize"] = [9,5]
# plt.rcParams.update({'font.size': 12})
# cf_matrix= confusion_matrix(vy,y_pred)
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = ['{0:0.0f}'.format(value) for value in
#                 cf_matrix.flatten()]
# labels = [f"{v1}\n{v2}" for v1, v2 in
#           zip(group_names,group_counts)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cf_matrix, annot=labels, fmt='',)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.savefig("Confusion Matrix.png")
# plt.show()

