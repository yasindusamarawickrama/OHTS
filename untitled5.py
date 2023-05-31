import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

df = pd.read_csv(r'C:/Users/yasin/Desktop/OHTS Assignment/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')

df.drop(columns = [' Fwd Header Length'])
df.fillna(0)

from sklearn import preprocessing

# label_encoder = preprocessing.LabelEncoder()
# df['Flow ID']= label_encoder.fit_transform(df['Flow ID'])
# df['Flow ID'].unique()

# label_encoder = preprocessing.LabelEncoder()
# df[' Timestamp']= label_encoder.fit_transform(df[' Timestamp'])
# df[' Timestamp'].unique()

# label_encoder = preprocessing.LabelEncoder()
# df[' Source IP']= label_encoder.fit_transform(df[' Source IP'])
# df[' Source IP'].unique()

# label_encoder = preprocessing.LabelEncoder()
# df[' Destination IP']= label_encoder.fit_transform(df[' Destination IP'])
# df[' Destination IP'].unique()

from sklearn.preprocessing import LabelEncoder

def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df

df = Encoder(df)

features = ['Total Length of Fwd Packets',' Fwd Packet Length Max',' Flow IAT Mean',' Total Length of Bwd Packets']
X = df.loc[:, features]
y = df.loc[:, [' Label']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.2)

y_test

# from sklearn.neighbors import KNeighborsClassifier

# knn5 = KNeighborsClassifier(n_neighbors = 5)
# knn1 = KNeighborsClassifier(n_neighbors=1)

# knn5.fit(X_train, y_train)
# knn1.fit(X_train, y_train)

# y_pred_5 = knn5.predict(X_test)
# y_pred_1 = knn1.predict(X_test)

# from sklearn.metrics import accuracy_score

# print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
# print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

