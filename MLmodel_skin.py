import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

"""
This file is used to load the data and train the model for detecting melanoma.
"""

#load and process data
path = os.listdir("./train_skin")
id = {'benign':0,'malignant':1}
images, ids = [], []
for folder in id:
    path2 = "./train_skin" + "/" + folder
    for item in os.listdir(path2):
        img = cv2.imread(path2 + '/' + item, 0)
        img = cv2.resize(img, (200,200))
        images.append(img)
        ids.append(id[folder])
images = np.array(images)
ids = np.array(ids)
images = images.reshape(len(images), -1)
x_train, x_test, y_train, y_test = train_test_split(images, ids, random_state=10,
                                                    test_size=.2)

#feature scaling
x_train = x_train / 255
x_test = x_test / 255

#train
svc = SVC()
svc.fit(x_train,y_train)

#evaluate
print("Training score:", svc.score(x_train, y_train))
print("Testing Score:", svc.score(x_test, y_test))

#test
decode = {0 : 'benign', 1 : "malignant"}
prediction, correct= [], []
for i in os.listdir("./test_skin/benign/"):
    img = cv2.imread("./test_skin/benign/" +i,0)
    img2 = cv2.resize(img, (200,200))
    img2 = img2.reshape(1,-1) / 255
    p = svc.predict(img2)
    prediction.append(p[0])
    correct.append(0)

for i in os.listdir('./test_skin/malignant/'):
    img = cv2.imread('./test_skin/malignant/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = svc.predict(img1)
    prediction.append(p[0])
    correct.append(1)

print("accuracy score:", accuracy_score(prediction, correct))

#uncomment to save modek as a pkl file
#pickle.dump(svc, open('detect_melanoma.pkl','wb'))