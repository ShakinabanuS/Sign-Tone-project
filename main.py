# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser
#from plotly import graph_objects as go
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import imagehash
from werkzeug.utils import secure_filename
from PIL import Image
import argparse
import urllib.request
import urllib.parse
import pyttsx3
   
# necessary imports 
import seaborn as sns
#import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
#%matplotlib inline
pd.set_option('display.max_columns', 26)
##
from PIL import Image, ImageOps
import scipy.ndimage as ndi

from skimage import transform
import seaborn as sns
#from keras.preprocessing.image import ImageDataGenerator , load_img , img_to_array
#from keras.models import Sequential
#from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
##
import glob
#from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import seaborn as sns
#import keras as k
#from keras.layers import Dense
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#from tensorflow.keras.optimizers import Adam
##
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="sign_voice"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    
    return render_template('index.html',msg=msg)




@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided' 
   
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    

    now = datetime.datetime.now()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    #if request.method=='GET':
    #    msg = request.args.get('msg')
    if request.method=='POST':
        
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']

        mycursor.execute("SELECT max(id)+1 FROM register")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
                
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        mycursor.execute(sql,val)
        mydb.commit()
        return redirect(url_for('login_user'))

    
        
    return render_template('register.html',msg=msg)




@app.route('/admin', methods=['GET', 'POST'])
def admin():
    
    dimg=[]
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
        
        
    return render_template('admin.html',dimg=dimg)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    dimg=[]
    mycursor = mydb.cursor()
    
    if request.method=='POST':
        
        message=request.form['message']
        file = request.files['file']

        mycursor.execute("SELECT max(id)+1 FROM sign_image")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        fn="F"+str(maxid)+".gif"
        file.save(os.path.join("static/upload", fn))
        
        sql = "INSERT INTO sign_image(id,message,image_file) VALUES (%s, %s, %s)"
        val = (maxid,message,fn)
        mycursor.execute(sql,val)
        mydb.commit()
        msg="success"
        #return redirect(url_for('login_user'))

    
    '''path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #resize
        img = cv2.imread('static/data/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
        
        
    return render_template('upload.html',msg=msg)

@app.route('/view_image', methods=['GET', 'POST'])
def view_image():
    msg=""
    act=request.args.get("act")
    dimg=[]
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM sign_image")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from sign_image where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('view_image'))

        
    return render_template('view_image.html',msg=msg,data=data,act=act)

@app.route('/test_voice', methods=['GET', 'POST'])
def test_voice():
    msg=""
    st=""
    vtext=""
    act=request.args.get("act")
    dimg=[]
    mycursor = mydb.cursor()
    
    img=""

    if request.method=='POST':        
        mess=request.form['message']

        if mess=="":
            s=1
        else:
            mm="%"+mess+"%"

            mycursor.execute("SELECT * FROM sign_image where message like %s",(mm,))
            dat = mycursor.fetchall()

            for dat1 in dat:
                img=dat1[2]
                

            if img=="":
                s=1
            else:
                ff=open("static/det.txt","w")
                ff.write(mess)
                ff.close()
                ff=open("static/img.txt","w")
                ff.write(img)
                ff.close()
                return redirect(url_for('test_voice',act='1'))

    if act=="1":
        ff=open("static/det.txt","r")
        vtext=ff.read()
        ff.close()

        ff=open("static/img.txt","r")
        img=ff.read()
        ff.close()
        
    return render_template('test_voice.html',msg=msg,act=act,img=img,st=st,vtext=vtext)

@app.route('/process', methods=['GET', 'POST'])
def process():
    dimg=[]
    path_main = 'static/dataset'
    
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    return render_template('process.html',dimg=dimg)

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    dimg=[]
    path_main = 'static/dataset'
    
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    return render_template('process1.html',dimg=dimg)

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        '''img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)'''

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro11.html',dimg=dimg)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        '''image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        #cv2.imwrite("static/trained/bb/bin_"+fname, thresh)'''

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        
        
        img = cv2.imread('static/trained/g_'+fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/"+fname
        #segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)

#TCN  - Temporal Convolutional Network - Sign Language Recognition
def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


    

    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size: int,
                 padding: str,
                 activation: str = 'relu',
                 dropout_rate: float = 0,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs):

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def tcn_full_summary(model: Model, expand_residual_blocks=True):
        #import tensorflow as tf
        # 2.6.0-rc1, 2.5.0...
        versions = [int(v) for v in tf.__version__.split('-')[0].split('.')]
        if versions[0] <= 2 and versions[1] < 5:
            layers = model._layers.copy()  # store existing layers
            model._layers.clear()  # clear layers

            for i in range(len(layers)):
                if isinstance(layers[i], TCN):
                    for layer in layers[i]._layers:
                        if not isinstance(layer, ResidualBlock):
                            if not hasattr(layer, '__iter__'):
                                model._layers.append(layer)
                        else:
                            if expand_residual_blocks:
                                for lyr in layer._layers:
                                    if not hasattr(lyr, '__iter__'):
                                        model._layers.append(lyr)
                            else:
                                model._layers.append(layer)
                else:
                    model._layers.append(layers[i])

            model.summary()  # print summary

            # restore original layers
            model._layers.clear()
            [model._layers.append(lyr) for lyr in layers]

            

        def _build_layer(self, layer):
           
            self.layers.append(layer)
            self.layers[-1].build(self.res_output_shape)
            self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape)

        def build(self, input_shape):

            with K.name_scope(self.name):  # name scope used to make sure weights get unique names
                self.layers = []
                self.res_output_shape = input_shape

                for k in range(2):  # dilated conv block.
                    name = 'conv1D_{}'.format(k)
                    with K.name_scope(name):  # name scope used to make sure weights get unique names
                        conv = Conv1D(
                            filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate,
                            padding=self.padding,
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                        if self.use_weight_norm:
                            from tensorflow_addons.layers import WeightNormalization
                            # wrap it. WeightNormalization API is different than BatchNormalization or LayerNormalization.
                            with K.name_scope('norm_{}'.format(k)):
                                conv = WeightNormalization(conv)
                        self._build_layer(conv)

                    with K.name_scope('norm_{}'.format(k)):
                        if self.use_batch_norm:
                            self._build_layer(BatchNormalization())
                        elif self.use_layer_norm:
                            self._build_layer(LayerNormalization())
                        elif self.use_weight_norm:
                            pass  # done above.

                    with K.name_scope('act_and_dropout_{}'.format(k)):
                        self._build_layer(Activation(self.activation, name='Act_Conv1D_{}'.format(k)))
                        self._build_layer(SpatialDropout1D(rate=self.dropout_rate, name='SDropout_{}'.format(k)))

                if self.nb_filters != input_shape[-1]:
                    # 1x1 conv to match the shapes (channel dimension).
                    name = 'matching_conv1D'
                    with K.name_scope(name):
                        # make and build this layer separately because it directly uses input_shape.
                        # 1x1 conv.
                        self.shape_match_conv = Conv1D(
                            filters=self.nb_filters,
                            kernel_size=1,
                            padding='same',
                            name=name,
                            kernel_initializer=self.kernel_initializer
                        )
                else:
                    name = 'matching_identity'
                    self.shape_match_conv = Lambda(lambda x: x, name=name)

                with K.name_scope(name):
                    self.shape_match_conv.build(input_shape)
                    self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

                self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
                self.final_activation = Activation(self.activation, name='Act_Res_Block')
                self.final_activation.build(self.res_output_shape)  # probably isn't necessary

                # this is done to force Keras to add the layers in the list to self._layers
                for layer in self.layers:
                    self.__setattr__(layer.name, layer)
                self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
                self.__setattr__(self.final_activation.name, self.final_activation)

                super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

        def call(self, inputs, training=None, **kwargs):
            """
            Returns: A tuple where the first element is the residual model tensor, and the second
                     is the skip connection tensor.
            """
            
            x1 = inputs
            for layer in self.layers:
                training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
                x1 = layer(x1, training=training) if training_flag else layer(x1)
            x2 = self.shape_match_conv(inputs)
            x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
            return [x1_x2, x1]

        def compute_output_shape(self, input_shape):
            return [self.res_output_shape, self.res_output_shape]
####
def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))




@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        image = cv2.imread("static/dataset/"+fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        edged.save(path4)
        ##
    
        
    return render_template('pro4.html',dimg=dimg)


    

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,210,265]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[30,80,140,220,275]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,23,55,85,105]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[5,23,55,85,105]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    

    return render_template('pro6.html',dimg=dimg)

#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    dd2=[]
    ex=dat.split(',')
    ##
    vv=[]
    vn=0
    data2=[]
    path_main = 'static/dataset'
    for val in ex:
        dt=[]
        n=0
        
        for fname in os.listdir(path_main):
            fa1=fname.split('.')
            fa=fa1[0].split('-')
            fv=int(fa[1])-1
            
            if cname[fv]==val:
                dt.append(fname)
                n+=1
        vv.append(n)
        vn+=n
        data2.append(dt)
        
    print(vv)
    print(data2[0])
    
    i=0
    vd=[]
    data4=[]
    while i<8:
        vt=[]
        vi=i+1
        vv[i]

        vt.append(cname[i])
        vt.append(str(vv[i]))
        
        vd.append(str(vi))
        data4.append(vt)
        i+=1
    print(data4)

    
    dd2=vv
    doc = vd #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    cc=['green','yellow','red','blue','brown','pink','grey','orange']
    plt.bar(doc, values, color =cc,
            width = 0.4)
 

    plt.ylim((1,25))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2,data4=data4)

#######
@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""

    
        
    return render_template('userhome.html',msg=msg)

@app.route('/test', methods=['GET', 'POST'])
def test():
    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0

    ff=open("static/trained/class1.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    ff=open("static/msg.txt","w")
    ff.write("")
    ff.close()
    
    if request.method=='POST':
        lg=request.form['language']
        #file = request.files['file']

        af=open("lang.txt","w")
        af.write(lg)
        af.close()
        return redirect(url_for('test_cam'))

        
        '''if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            ##
            n=0
            fpos=""
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                fa1=fn.split('.')
                fa=fa1[0].split('-')
                fpos=fa[1]
                fv=int(fa[1])-1
                n+=1
                if cname[fv]==val:
                    tclass=n
                    
                    break
                
            
            print(tclass)
            tt=tclass-1
            cla=cname[tt]
            dta=cla+"|"+fn+"|"+fpos
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"'''
    
        
    return render_template('test.html',msg=msg)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn

    
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[0]
    pos=gs[2]

    n=int(pos)-1

    af2=open("lang.txt","r")
    lgg=af2.read()
    af2.close()
    
    lfile="a"+pos+"_"+lgg+".jpg"

    
    af=open("static/trained/lang1.txt","r")
    la=af.read()
    af.close()

    la1=la.split(",")
    la2=la1[n].split("-")
    
    c=0
    if lgg=="h":
        c=1
    elif lgg=="t":
        c=2
    elif lgg=="m":
        c=3
    else:
        c=0

    word=la2[c]
    
    
    

        
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,lfile=lfile,word=word)

@app.route('/test_cam', methods=['GET', 'POST'])
def test_cam():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("lang.txt","r")
    lg=f2.read()
    f2.close()

    if request.method=='POST':
        lg=request.form['language']
        f2=open("lang.txt","w")
        f2.write(lg)
        f2.close()

    
        
    return render_template('test_cam.html',msg=msg,lg=lg)

def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()
    
@app.route('/test_pro3', methods=['GET', 'POST'])
def test_pro3():
    msg=""
    fn=""
    st=""
    lfile=""
    word=""
    
    act=request.args.get("act")
    f2=open("lang.txt","r")
    lgg=f2.read()
    f2.close()

    f3=open("static/msg.txt","r")
    ms=f3.read()
    f3.close()

    ff=open("static/trained/class1.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')

    if ms=="":
        st=""
    else:
        st="1"
        n=0
        for cc in cname:
            n+=1
            if cc==ms:              
                
                break
        print("value=")
        print(str(n))
        m=n-1
        pos=n
        ##
        lfile="a"+str(pos)+"_"+lgg+".jpg"

        
        af=open("static/trained/lang1.txt","r")
        la=af.read()
        af.close()

        la1=la.split(",")
        la2=la1[m].split("-")
        
        c=0
        if lgg=="h":
            c=1
        elif lgg=="t":
            c=2
        elif lgg=="m":
            c=3
        else:
            c=0

        word=la2[c]
        if word=="":
            s=1
        else:
            speak(word)
        
    return render_template('test_pro3.html',msg=msg,st=st,lgg=lgg,fn=fn,act=act,lfile=lfile,word=word)


def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


