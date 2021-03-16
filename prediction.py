# To load dataset into pandas dataframe
import pandas as pd

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings('ignore')

# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Correlation Matrix with Heatmap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

#  SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import sklearn.metrics as sm

# Keras Deep learning library
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow import keras



# Function will receiver file name list, path to files, and exchnage name as parameter,
# Basic objective ti to laod the files.

def Read_Dataset(file_name_list, path , exchange):
    
    # empty dataframe
    df = pd.DataFrame()
    
    #iterate over the file name list
    for files in file_name_list:
        
        # read files one by one
        df_tmp = pd.read_csv(path + files)
        
        # update empty dataframe by each file, step by step
        df = df.append(df_tmp)
        
    # add exchange columns to final dataframe (combination of 4 files for each exchange)
    df["Exchange"] = exchange
    
    # Convert date object into numeric
    df["receiveTs"] = pd.to_datetime(df["receiveTs"]).dt.strftime("%Y%m%d %H:%M:%S")
    df['receiveTs'] = pd.to_datetime(df['receiveTs'])
    
    # Calculate midprice as  midPrice = (Pa_1 + Pb_1) / 2
    df['Mid_Price'] = (df['Pa_1'] + df['Pb_1'] ) / 2
    return df
        
def Resample_Data(df,freq):
    df = df.resample(freq, on="receiveTs").mean()
    return df


def Convert_Date_Split(df):
    df['year'] = df['receiveTs'].dt.year
    df['month'] = df['receiveTs'].dt.month
    df['day'] = df['receiveTs'].dt.day
    df['hour'] = df['receiveTs'].dt.hour
    df['minute'] = df['receiveTs'].dt.minute
    df['second'] = df['receiveTs'].dt.second
    
    df.drop(['receiveTs'], axis=1, inplace=True)
    return df

def main():

    # Exchange A data files, if you have the extracted the folder, 
    # you can update 'path' variable with your working directory.
    path = "./Data/Price Prediction/"

    exchange_A = ["exchange-a-orderbook 2020-08-01.csv", "exchange-a-orderbook 2020-08-02.csv",
                "exchange-a-orderbook 2020-08-15.csv","exchange-a-orderbook 2020-08-16.csv"]

    exchange_B = ["exchange-b-orderbook 2020-08-01.csv", "exchange-b-orderbook 2020-08-02.csv",
                "exchange-b-orderbook 2020-08-15.csv","exchange-b-orderbook 2020-08-16.csv"]

    exchange_C = ["exchange-c-orderbook 2020-08-01.csv", "exchange-c-orderbook 2020-08-02.csv",
                "exchange-c-orderbook 2020-08-15.csv","exchange-c-orderbook 2020-08-16.csv"]



    #Preprocessing 

    # Load exchange a files
    df_A = Read_Dataset(exchange_A, path, 1)
    print(df_A.shape)

    # Load exchange b files
    df_B = Read_Dataset(exchange_B, path, 2)
    print(df_B.shape)


    # Load exchange c files
    df_C = Read_Dataset(exchange_C, path, 3)
    print(df_C.shape)


    # Create a single master dataframe to merge b and c dataframe.

    df_train = pd.DataFrame()
    df_train = df_train.append(df_B)
    df_train = df_train.append(df_C)
    print(df_train.shape)


    # CHeck Null values in df_A for exchange a
    df_A.isnull().sum()

    # CHeck Null values in df_train for exchange b and c
    df_train.isnull().sum()


    # dataset summary
    df_train.describe()

    # Here we are resampling data on 5 seconds frequency, so the prediciton will be on 5s based.
    df_A = Resample_Data(df_A , "5S")
    df_train = Resample_Data(df_train , "5S")

    # After resampling, we need to reset the index.
    df_A = df_A.reset_index()
    df_train = df_train.reset_index()


    # Drop time column to convert other  clumn to int
    time_a = df_A['receiveTs']
    df_A.drop(['receiveTs'], axis= 1 , inplace =True)

    time_train = df_train['receiveTs']
    df_train.drop(['receiveTs'], axis= 1 , inplace =True)


    df_train.dropna(inplace=True)
    df_A.dropna(inplace=True)


    #Feature Importance
    # Split into training dataset
    X_train = df_A.drop(['Mid_Price'], axis=1)
    y_train = df_A['Mid_Price']

    # Create model object and train on the training data to get relevant features
    dtree = DecisionTreeRegressor(random_state = 2, min_samples_leaf = 5, min_samples_split =20)
    dtree.fit(X_train,y_train)


    # Extract the feature based on their importance
    df_DTree = pd.DataFrame({'importance' : dtree.feature_importances_}, index = X_train.columns).sort_values('importance', ascending = False)
    columns_list = df_DTree.index
    columns_list = columns_list[0:20]
    print ("Top 20 most relvant and important features.")
    list(columns_list)


    # Update both dataframe to use only relevant feature w.r.t target column.
    df_A = df_A[list(columns_list)]
    df_A['Mid_Price'] = y_train
    # adding time column back
    df_A['receiveTs'] = time_a

    mid_price_train = df_train['Mid_Price']
    df_train = df_train[list(columns_list)]
    df_train['Mid_Price'] = mid_price_train
    # adding time column back
    df_train['receiveTs'] = time_train


    df_A['receiveTs'] = time_a
    df_train['receiveTs'] = time_train

    df_A = Convert_Date_Split(df_A)

    df_train = Convert_Date_Split(df_train)

    #Model Implementation

    # Split Exchange b an c for model training.
    X = df_train.drop('Mid_Price',1)
    y = df_train['Mid_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=50)
    print ("Training data shape is:")
    print (X_train.shape)
    print (y_train.shape)

    X1 = df_A.drop('Mid_Price',1)
    y1 = df_A['Mid_Price']
    print ("Testing data shape is:")
    print (X1.shape)
    print (y1.shape)


    # Training model on training data (Exchange b , c)
    reg = LinearRegression()
    reg.fit(X_train , y_train)


    # Making prediction on testing data (Exchange a)
    y_pred1 = reg.predict(X1)

    # Model Accuracy
    print ("Model Error Report")
    print("Mean absolute error =", (sm.mean_absolute_error(y1, y_pred1))) 
    print("Mean squared error =", (sm.mean_squared_error(y1, y_pred1))) 
    print("Median absolute error =", (sm.median_absolute_error(y1, y_pred1))) 
    print("Explain variance score =", (sm.explained_variance_score(y1, y_pred1))) 
    print ("\n")
    print ("Model Accuracy")
    print("R2 score =", sm.r2_score(y1, y_pred1))




    # Training model on training data (Exchange b , c)
    DT_regr = DecisionTreeRegressor(max_depth=20)
    DT_regr.fit(X_train, y_train)

    # Making prediction on testing data (Exchange a)
    pred_2 = DT_regr.predict(X1)

    # Model Accuracy

    print ("Model Error Report")
    print("Mean absolute error =", (sm.mean_absolute_error(y1, pred_2))) 
    print("Mean squared error =", (sm.mean_squared_error(y1, pred_2))) 
    print("Median absolute error =", (sm.median_absolute_error(y1, pred_2))) 
    print("Explain variance score =", (sm.explained_variance_score(y1, pred_2)))
    print ("\n")
    print ("Model Accuracy")
    print("R2 score =", (sm.r2_score(y1, pred_2)))



    # Initialising the ANN
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(32, activation = 'relu', input_dim = X_train.shape[1]))

    # Adding the second hidden layer
    model.add(Dense(units = 32, activation = 'relu'))

    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = 'relu'))

    # Adding the output layer

    model.add(Dense(units = 1))

    # Compiling the ANN
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile( optimizer=opt, loss = 'mean_squared_error')

    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size = 50, epochs = 10)

    y_pred3 = model.predict(X1)


    # Model Accuracy

    print("Mean absolute error =", (sm.mean_absolute_error(y1, y_pred3))) 
    print("Mean squared error =", (sm.mean_squared_error(y1, y_pred3))) 
    print("Median absolute error =", (sm.median_absolute_error(y1, y_pred3))) 
    print("Explain variance score =", (sm.explained_variance_score(y1, y_pred3))) 
    print("R2 score =", (sm.r2_score(y1, y_pred3)))


    X1['l_predictions'] = y_pred1
    X1['nn_predictions'] = y_pred3
    X1.to_csv('./predictions.csv')

if __name__ == '__main__':
    main()
