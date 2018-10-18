from __future__ import print_function
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
#************************************************************************#
TrainData = pd.read_csv('train-v3.csv')
ValidData = pd.read_csv('valid-v3.csv')
TestData = pd.read_csv('test-v3.csv')


TotalTrainData = ["sale_yr","sale_month","sale_day","bedrooms",
                 "bathrooms","sqft_living","sqft_lot","floors","waterfront",
                 "view","condition","grade","sqft_above","sqft_basement",
                 "yr_built","yr_renovated","zipcode","lat","long",
                 "sqft_living15","sqft_lot15"]

x_Train = TrainData[TotalTrainData]
y_Train = TrainData['price']
x_Valid = ValidData[TotalTrainData]
y_Valid = ValidData['price']

#print(TrainData['price'].describe())
#sns.distplot(TrainData['price']) #畫圖
#print("Skewness: %f" % TrainData['price'].skew()) #計算偏度
#print("Kurtosis: %f" % TrainData['price'].kurt()) #計算峰度

#sns.pairplot(TrainData[c],size=2) #畫資料線性圖
#*****************關係矩陣，熱像圖*************
#corrmat = TrainData.corr()
#f,ax = plt.subplots(figsize=(10,10))
#sns.heatmap(corrmat,vmax=0.8,square=True)
#*******************************************
#**********************正規化**************************************#
def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)

def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c]-mu[c])/s[c]
    return df

stats = norm_stats(x_Train, x_Valid)
arr_x_train = np.array(z_score(x_Train, stats))
arr_y_train = np.array(y_Train)
arr_x_valid = np.array(z_score(x_Valid, stats))
arr_y_valid = np.array(y_Valid)

print('Training shape:', arr_x_train.shape)
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])
#**********************結束**************************************#
def basic_model():   
    model = Sequential() #序貫模型
    model.add(Dense(300, activation="tanh", kernel_initializer='normal', input_dim=arr_x_train.shape[1]))#將layer增加到模型
    model.add(Dropout(0.1))
    model.add(Dense(200, activation="tanh", kernel_initializer='normal', 
        kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(125, activation="relu", kernel_initializer='normal',                   
        kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    model.add(Dropout(0.05))
    model.add(Dense(5, activation="linear", kernel_initializer='normal'))
    model.add(Dropout(0.0))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mae])
    return model 
    

model = basic_model()
model.summary()   

epochs = 500
batch_size = 128

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

history = model.fit(arr_x_train,arr_y_train,batch_size=batch_size,epochs=epochs,validation_data=(arr_x_valid, arr_y_valid))
#**********************畫圖**************************************#
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
#**********************結束**************************************#
#**********************預測**************************************#
x_test = TestData[TotalTrainData].values
TestData['lat'] = np.log1p(TestData['lat'])
TestData = pd.get_dummies(TestData)
scale = StandardScaler()
x_test = scale.fit_transform(x_test)
prediction = model.predict(x_test)
prediction = pd.DataFrame(prediction, columns=['price'])
result = pd.concat([TestData ['id'], prediction], axis=1)
result.columns
result.to_csv('./Predictions.csv', index=False)
#**********************結束**************************************#