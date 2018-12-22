#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D
import pickle,time
from tensorflow.keras.callbacks import TensorBoard
##import matplotlib.pyplot as plt
#import numpy as np
#import os,random,pickle
#import cv2

#dir_name = "C:\Info\Dogs vs Cats\PetImages"
Categories = ['Dog','Cat']
#
#training_data = []
#
#def create_training_data():
#    for category in Categories:
#        path = os.path.join(dir_name,category)
#        class_num = Categories.index(category)
#        for img in os.listdir(path):
#            try:
#                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
#                img_size = 50
#                new_array  = cv2.resize(img_array,(img_size,img_size))
#                training_data.append([new_array,class_num])
#            except Exception as e:
#                pass
#    random.shuffle(training_data)
#    
#    X,y = [],[]
#    for features,labels in training_data:
#        X.append(features)
#        y.append(labels)
#        
#    X = np.array(X).reshape(-1,img_size,img_size,1)
#    
#    pickle_out = open('X.pickle','wb')
#    pickle.dump(X,pickle_out)
#    pickle_out.close()
#    
#    pickle_out = open('y.pickle','wb')
#    pickle.dump(y,pickle_out)
#    pickle_out.close()   
#    
#
#create_training_data() 


name = "Cats vs Dogs-cnn-2x64-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir=f'logs/{name}')
 
pickle_in_X = open('X.pickle','rb')
X = pickle.load(pickle_in_X)

pickle_in_y = open('y.pickle','rb')
y = pickle.load(pickle_in_y)

X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=5,validation_split=0.1,callbacks=[tensorboard])
model.save('Cats_v_Dogs.model')

#def predict_img(filepath):
#    img_size = 50
#    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
#    new_array  = cv2.resize(img_array,(img_size,img_size))
#    return new_array.reshape(-1,img_size,img_size,1)
#
#model = tf.keras.models.load_model('Cats_v_Dogs.model')
#
#prediction = model.predict([predict_img('dog.jpg')])
#
#print(Categories[int(prediction[0][0])])





