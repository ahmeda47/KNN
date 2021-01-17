import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

#create object to take labels and encode them into integer values
le = preprocessing.LabelEncoder()

#create list for each label
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

#create X and Y lists
X = list(zip(buying, maint,doors,persons, lug_boot, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

print(x_train, y_test)

#implement the model
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)

#where the data points are, what are the prdictions, and actual values
predicted = model.predict(x_test)

names = ["unacc","acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N:", n)