from sklearn.neighbors import KNeighborsClassifier

model = knn1
model = joblib.load("/content/port_scanner.sav")

input_data = [[2664, 456, 15075.500000, 6954], [0, 0, 38.500000, 0], [46, 46 ,101680.000000, 253]]

predictions = model.predict(input_data)
print(predictions)
