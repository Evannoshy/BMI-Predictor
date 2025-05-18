#import ext lib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt
import seaborn as sns


#start of code for model 
#reading csv file
df = pd.read_csv(r'C:\Users\User\Desktop\python\BMI.csv', encoding='utf-8-sig')
df.head()

#START OF DATA PROCESSING

#first line to print which columns there are, second line to remove any start/end spaces,
#third line to specify the first line is header
#print(df.columns.tolist())
df.columns = df.columns.str.strip()
df = pd.read_csv('BMI.csv', header=0)

#Since KNN algo requires numerical input, I needed to convert the Gender into numerical format. 
#Using label encoding, I chose to label Male as 0 and Female as 1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
#END OF DATA PREPROCESSING

#START OF MODELLING
#define variables/axis, x -> features, y->target variable. Since we are predicting BMI, it shd be y.  
X = df[['Height', 'Weight', 'Gender']] 
y = df['Index']

#with the random state parameter, python will shuffle before splitting. Hence, set to specific integer value 
#thus split the same way everytime the code runs. Test size -> % of data tested
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 23, test_size=0.3)

#create a min-max scaler and apply it to x. This rescales each feature to the range, sinc eKNN is distance based
scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#create a KNN classifier. .fit() stores training points internally. After training, 
#.predict() applies the KNN model to the test features
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test)
knn.score(X_test, y_test)
#END OF MODELLING

#RESULT EVALUATION
#confusion matrix shows counts of correct vs incorrect classification by class.
#true negative is on the top left, false positive is on the top right, false negative is at the bottom left, true positive is at the bottom right
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred)

#classification report shows precision, recall and f1 scores
cr = classification_report(y_test, y_pred)
#print(cm)

#DATA VISUALISATION

bmi_labels = {
    0: "Underweight",
    1: "Normal",
    2: "Overweight",
    3: "Obese",
    4: "Extremely Obese"
}

'''
sns.countplot(x='Index', data=df)
plt.title('BMI Class Distribution')
plt.xlabel('BMI Category (Index)')
plt.ylabel('Count')
plt.show()
'''


plt.figure(figsize=(8,6))
sns.scatterplot(x='Height', y='Weight', hue='Index', style='Gender', data=df, palette='Set2')
plt.title('Height vs Weight Colored by BMI and Gender Shape')
plt.show()

'''
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
'''

'''
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Height'], df['Weight'], df['Gender'], c=df['Index'], cmap='twilight_r')
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Gender')
plt.title('3D Plot of Features Colored by BMI')
plt.show()
'''

#APPLICATION OF MODEL
#Now that I have the model, I can input unseen data into it for prediction
#Ensure that the data is in a height/weight/gender format
new_data = [[170, 4, 0]]

#New data has to be scaled as KNN is distance based
new_data_scaled = scaler.transform(new_data)
prediction = knn.predict(new_data_scaled)
print("Predicted BMI Index:", bmi_labels[prediction[0]])
