import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import numpy as np
import os,random,pickle,time
import cv2

#dir_name = os.path.dirname(__file__)

class cnn_mxn:
    def __init__(self,img_size,categories,dense_layers,conv_layers,epochs,node_no,batch_size,optimizer='adam',loss_algo='binary_crossentropy'):
        self.img_size = img_size
        self.categories = categories
        self.dense_layers = dense_layers
        self.conv_layers = conv_layers
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_algo = loss_algo
        self.node_no = node_no     
        self.batch_size = batch_size
    
    #Activate only if GPU version of tf is installed along with CUDA Toolkit for NVDIA  and if you want some portion of GPU to run    
    def factor_gpu(self,percent_gpu=0.5):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_functions=percent_gpu)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def create_training_data(self):
        training_data = []
        for category in self.categories:
            path = os.path.join(dir_name,category)
            class_num = self.categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                    new_array  = cv2.resize(img_array,(self.img_size,self.img_size))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
        random.shuffle(training_data)
         
        X,y = [],[]
        for features,labels in training_data:
            X.append(features)
            y.append(labels)
            
        X = np.array(X).reshape(-1,self.img_size,self.img_size,1)
        X = X/255.0 # normalize data(image normalization)
        return X,y
        
    def pickling_needed(self,X,y):
        with open('X.pickle','wb') as f:
            pickle.dump(X,f)
        with open('y.pickle','wb') as g:
            pickle.dump(y,g)
        
    def load_pickle(self):
        if os.path.exists('X.pickle'):
            with open('X.pickle','rb') as f:
                X = pickle.load(f)
            with open('y.pickle','rb') as g:
                y = pickle.load(g)
        return X,y
                
    def initialize_tensorboard(self,name):
        tensorboard = TensorBoard(log_dir=f'logs/{name}')        
        return tensorboard
                
    def define_model(self,X,y,initialize_tensorboard=False):        
        for dense_layer in range(1,self.dense_layers+1):
            for conv_layer in range(1,self.conv_layers+1):
                NAME = f'{conv_layer}-conv-{dense_layer}-dense-{int(time.time())}'
                
                model = Sequential()
                
                model.add(Conv2D(self.node_no,(3,3),input_shape=X.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                
                for l in range(conv_layer-1):
                    model.add(Conv2D(self.node_no,(3,3)))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2,2)))
            
                model.add(Flatten())
                
                for l in range(dense_layer):
                    model.add(Dense(self.node_no))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.1))
            
                model.add(Dense(1))
                model.add(Activation('softmax')) 
        
                model.compile(loss=self.loss_algo,optimizer=self.optimizer,metrics=['accuracy']) 
        
                if initialize_tensorboard: 
                    tensorboard = self.initialize_tensorboard(NAME)
                    checkpoint = self.check_point()
                    model.fit(X,y,batch_size=self.batch_size,epochs=self.epochs,validation_split=0.15,callbacks=[tensorboard,checkpoint])
                else:
                    model.fit(X,y,batch_size=self.batch_size,epochs=self.batch_size,validation_split=0.15)                    
                model.save(NAME+'.model')
                    
    def check_point(self):
        filepath = "CNN-{epoch:02d}-{val_acc:.3f}"
        checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))
        return checkpoint  
    
    def predict_img(self,filename):
        img_array = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        new_array  = cv2.resize(img_array,(self.img_size,self.img_size))
        return new_array.reshape(-1,self.img_size,self.img_size,1)    
    
    #check for best model and run that model    
    def run_model(self,name,filename,initialize_tensorboard=True):
        if initialize_tensorboard:
            model = tf.keras.models.load_model(name)    
            prediction = model.predict([self.predict_img(filename)]) 
    
        return self.categories[int(prediction[0][0])]             
    
Categories = ['Dog','Cat']
dir_name = "C:\Info\Deep learning App\PetImages\logs"
training = cnn_mxn(50,Categories,1,3,10,64,32)
#X,y = training.create_training_data()
#training.pickling_needed(X,y)
#X,y = training.load_pickle()
#training.define_model(X,y,initialize_tensorboard=True)

for file in os.listdir('./'):
    if file.endswith('.model'):
        prediction = training.run_model(file,'cat.jpg',initialize_tensorboard=True)
        print(file,prediction)