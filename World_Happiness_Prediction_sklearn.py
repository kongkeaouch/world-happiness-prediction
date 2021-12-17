import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import drive

drive.mount("/content/drive")
demography = pd.read_csv("drive/kongkea/Dataset/demography.csv")
demography = pd.DataFrame(happy)
demography.head()
demography.shape
demography.isnull().sum()
demography.describe()
sns.boxplot(demography["Happiness Rank"],
            demography["Health (Life Expectancy)"])
fig = plt.figure(figsize=(10, 8))
data = [
    demography["Life Expectancy"],
    demography["Freedom"],
    demography["Government Corruption"],
    demography["Dystopia Residual"],
]
plt.boxplot(data)
plt.show()
demography.head()
y = demography["Happiness Rank"]
fin_data = demography.drop(["Region", "Country", "Happiness Rank"], axis=1)
fin_data.columns
scale = StandardScaler()
data = scale.fit_transform(fin_data)

pickle.dump(scale, open("drive/kongkea/Dataset/HappinessScaler.pickle", "wb"))
print(data[0][0])
fig = plt.figure(figsize=(10, 8))
inserting = [data[4], data[5], data[7]]
plt.boxplot(inserting)
plt.show()
Xtrain, Xtest, ytrain, ytest = train_test_split(data, y, test_size=0.1)
reg = LinearRegression()
reg.get_params(deep=True)
vari = reg.fit(Xtrain, ytrain)
print(reg.score(Xtrain, ytrain))
pred = reg.predict(Xtest)
r2_score = reg.score(Xtest, ytest)
print(r2_score)
saved_model = pickle.dump(
    reg, open("drive/kongkea/Dataset/happy_model.pickle", "wb")
)
