import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier

# reading the training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# converting string labels to integer labels for training data which can be used for training the model
label_map = {
    'Apple_Raw': 0,
    'Apple_Ripe': 1,
    'Banana_Raw': 2,
    'Banana_Ripe': 3,
    'Coconut_Raw': 4,
    'Coconut_Ripe': 5,
    'Guava_Raw': 6,
    'Guava_Ripe': 7,
    'Leeche_Raw': 8,
    'Leeche_Ripe': 9,
    'Mango_Raw': 10,
    'Mango_Ripe': 11,
    'Orange_Raw': 12,
    'Orange_Ripe': 13,
    'Papaya_Raw': 14,
    'Papaya_Ripe': 15,
    'Pomengranate_Raw': 16,
    'Pomengranate_Ripe': 17,
    'Strawberry_Raw': 18,
    'Strawberry_Ripe': 19
}

# mapping the labels to the training data
train_data['category'] = train_data['category'].map(label_map)
train_data['category'] = train_data['category'].astype('int')

# dropping the ID column from the training and testing data
X = train_data.drop(["ID", "category"], axis=1)
y = train_data["category"]

# removing outliers using LOF algorithm
lof = LocalOutlierFactor(n_neighbors=40, contamination=0.01,metric = "euclidean")
y_pred = lof.fit_predict(X)
outliers = X[y_pred == -1]


# removing outliers from the training data
X = X[y_pred != -1]
y = y[y_pred != -1]

warnings.filterwarnings("ignore")

# splitting the training data into training and testing data
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:]

best_params_list = []

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# selecting the best features using PCA with  components
pca = PCA(n_components=359)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# selecting the best features using LDA with 19 components
lda = LDA(n_components=19)
X_train_pca = lda.fit_transform(X_train_pca, y_train)
X_test_pca = lda.transform(X_test_pca)

# using KMeans clustering to cluster the data into 20 clusters
n_clusters = 19
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_pca)

train_cluster_assignments = kmeans.predict(X_train_pca)
test_cluster_assignments = kmeans.predict(X_test_pca)

train_clusters = pd.DataFrame(train_cluster_assignments, columns=[19])
test_clusters = pd.DataFrame(test_cluster_assignments, columns=[19])

# adding the clusters to the training and testing data
X_train_pca = pd.concat([pd.DataFrame(X_train_pca), train_clusters], axis=1)
X_test_pca = pd.concat([pd.DataFrame(X_test_pca), test_clusters], axis=1)

# using Random Forest & Logistic Regression to train the model and predict the labels for the testing data
lr1 = RandomForestClassifier(n_estimators= 100)
lr2 = LogisticRegression(C= 0.01, dual= False, fit_intercept= True, max_iter= 500, penalty= 'l2', solver= 'newton-cg', warm_start= True)

# using Voting Classifier to combine the predictions of the 5 models
Voting = VotingClassifier(estimators=[('lr1', lr1), ('lr2', lr2)], voting='hard')
Voting.fit(X_train_pca, y_train)
pred1 = Voting.predict(X_test_pca)

# converting the integer labels to string labels
inv_label_map = {v: k for k, v in label_map.items()}
inv_y_pred = [inv_label_map[k] for k in pred1]

# saving the predictions to a csv file
y_pred = pd.DataFrame(inv_y_pred)
y_pred.columns = ['Category']
y_pred.to_csv('submission.csv', index=True)