import numpy as np
import pandas as pd
from sklearn import tree

# import dataset
irisDataset = pd.read_csv('Dataset Iris.csv', delimiter=';', header=0)

# ubah Species string menjadi int
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]

# hapus id
irisDataset = irisDataset.drop(labels="Id", axis=1)

# convert dataset ke numpy
irisDataset = irisDataset.to_numpy()

# split dataset
dataTraining = np.concatenate((irisDataset[0:40, :],
                               irisDataset[50:90, :]), axis=0)
dataTesting = np.concatenate((irisDataset[40:50, :],
                               irisDataset[90:100, :]), axis=0)
inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]

# define classifier
model = tree.DecisionTreeClassifier()

# train model
model = model.fit(inputTraining, labelTraining)

# predict
hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya ", labelTesting)
print("hasil prediksi :", hasilPrediksi)

# hitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, "data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah) * 100, "%")

# Bima Bayuaji - A11.2020.12731