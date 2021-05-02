from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=60)
print('data loaded')
print(faces.target_names)
print(faces.images.shape)

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# face data
# print(faces.images.shape)

# target values
# print(faces.target.shape)

# target value names
# print(faces.target_names)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(faces.images, faces.target, test_size=0.3, random_state=42)

x_train_1 = x_train.shape[0]
x_train_2 = x_train.shape[1]
x_train_3 = x_train.shape[2]

reshaped_x_train = x_train.reshape((x_train_1, x_train_2*x_train_3))

x_test_1 = x_test.shape[0]
x_test_2 = x_test.shape[1]
x_test_3 = x_test.shape[2]

reshaped_x_test = x_test.reshape((x_test_1, x_test_2*x_test_3))

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(model, {
    'svc__C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}, cv=3, return_train_score=False)

clf.fit(reshaped_x_train, y_train)
predictions = clf.best_estimator_.predict(reshaped_x_test)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)

# calculate precision, recall, fscore, and support
from sklearn.metrics import precision_recall_fscore_support

names = ['Rumsfeld', 'Bush', 'Schroeder']
score_labels = ['Recall', 'Precision', 'F1 Score', 'Support']
scores = precision_recall_fscore_support(y_test, predictions)
# index 0 is recall, index 1 is precision, index 2 is f1, index 3 is support

print("-------------------------------------------")
for i in range(len(scores)):
    for j in range(len(scores[i])):
        print(f"{names[j]} {score_labels[i]}: {scores[i][j]}")
    print("-------------------------------------------")

import matplotlib.pyplot as plt

# plot portraits
def plot_gallery(images, titles, h, w, n_row=4, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12, color="black") if "true" in titles[i] else plt.title(titles[i], size=12, color="red")
        plt.xticks(())
        plt.yticks(())
    plt.show(block=True)

# convert arrays of integers that represent a name to an array with the actual names themselves
def convert_index_to_names(data):  
    convert_to_name = lambda x: 'Rumsfeld' if x == 0 else ('Bush' if x == 1 else 'Schroeder')
    converted_arr = [convert_to_name(i) for i in data]
    return converted_arr

def title(y_pred, y_test, i):
    titles = []
    pred_name = convert_index_to_names(y_pred)
    true_name = convert_index_to_names(y_test)   

    for x in range(i):
        if y_pred[x] == y_test[x]: 
            titles.append('predicted: %s\ntrue:      %s' % (pred_name[x], true_name[x]))
        else:
            titles.append('predicted: %s\nfalse:      %s' % (pred_name[x], true_name[x]))
    
    return titles

prediction_titles = title(predictions, y_test, 24)
plot_gallery(x_test, prediction_titles, 62, 47)

import seaborn as sns 

x_axis_labels = ['Rumsfeld Actual', 'Bush Actual', 'Schroeder Actual']
y_axis_labels = ['Rumsfeld Predicted', 'Bush Predicted', 'Schroeder Predicted']
sns.heatmap(cm, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show(block=True)





