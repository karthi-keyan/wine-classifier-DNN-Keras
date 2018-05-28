#importing the required modules
import tensorflow as tf
import os
import numpy as np
from keras import Sequential,optimizers
from keras.layers import Dense,Dropout
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
os.chdir("C:/Users/karthi/Desktop/wine_dataset/")
#for defining the graph to visualize loss and accuracy
def get_graph(trn_res):
	fig,ax=plt.subplots(1,2)
	ax[0].set_title("Accuracy")
	ax[0].set_xlabel('epoch')
	ax[0].set_ylabel('accuracy')
	ax[0].plot(trn_res.history['acc'])
	ax[1].set_title("Loss")
	ax[1].set_xlabel('epoch')
	ax[1].set_ylabel('loss')
	ax[1].plot(trn_res.history['loss'])
	fig.canvas.draw()
	plt.savefig("training_result.png")
	plt.show()
#data preprocessing	
with open("wine_dataset.txt",'r') as file:
	data=file.readlines()
x,y=[],[]
for each in data: #data generation
	data_=[float(x.replace('\n','')) for x in each.split(',')]
	y.append(data_[0])
	x.append(np.array(data_[1:]).reshape((1,13)))
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.1)
y={1:np.array([1,0,0]),2:np.array([0,1,0]),3:np.array([0,0,1])}
trainx=np.array(trainx).reshape(160,1,13)
trainy=np.array([y[x] for x in trainy]).reshape(160,1,3)
testx=np.array(testx).reshape(18,1,13)
testy=np.array([y[x] for x in testy]).reshape(18,1,3)
#defining model
model=Sequential()
model.add(Dense(30,activation='sigmoid',input_shape=(1,13)))
model.add(Dense(10,activation='sigmoid'))
model.add(Dense(3,activation='sigmoid'))
opt=optimizers.Adam(lr=0.001)
#fitting and training
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
trn_res=model.fit(trainx,trainy,batch_size=10,nb_epoch=2000)
#testing data
score=model.evaluate(testx,testy)
print("test loss:{}  test accuracy:{}".format(score[0],score[1]))
#showing graph
get_graph(trn_res)