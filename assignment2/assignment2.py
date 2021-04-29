from sklearn.datasets import fetch_lfw_people

"""
def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images_shape)
"""
faces = fetch_lfw_people(min_faces_per_person=60)

from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

# face data
# print(faces.images)

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
model.fit(reshaped_x_train, y_train)

x_test_1 = x_test.shape[0]
x_test_2 = x_test.shape[1]
x_test_3 = x_test.shape[2]

reshaped_x_test = x_test.reshape((x_test_1, x_test_2*x_test_3))

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(svc, {
    'C': [1, 5, 10, 50],
    'gamma': ['auto', 'scale'] # may need to change later on
}, cv=3, return_train_score=False)

clf.fit(reshaped_x_train, y_train)

predictions = clf.best_estimator_.predict(reshaped_x_test)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)

# calculate average precision and average recall across three target values
def calculate_precision_and_recall(cm):
    donald_true_positives = cm[0][0]
    donald_false_positives = cm[0][1] + cm[0][2]
    donald_false_negatives = cm[1][0] + cm[2][0]
    donald_precision = donald_true_positives / (donald_true_positives + donald_false_positives)
    donald_recall = donald_true_positives / (donald_true_positives + donald_false_negatives)

    george_true_positives = cm[1][1]
    george_false_positives = cm[1][0] + cm[1][2]
    george_false_negatives = cm[0][1] + cm[2][1]
    george_precision = george_true_positives / (george_true_positives + george_false_positives)
    george_recall = george_true_positives / (george_true_positives + george_false_negatives)

    gerhard_true_positives = cm[2][2]
    gerhard_false_positives = cm[2][0] + cm[2][1]
    gerhard_false_negatives = cm[0][2] + cm[1][2]
    gerhard_precision = gerhard_true_positives / (gerhard_true_positives + gerhard_false_positives)
    gerhard_recall = gerhard_true_positives / (gerhard_true_positives + gerhard_false_negatives)

    avg_precision = (donald_precision + george_precision + gerhard_precision) / 3
    avg_recall = (donald_recall + george_recall + gerhard_recall) / 3

    return avg_precision, avg_recall

precision, recall = calculate_precision_and_recall(cm)

def calculate_f1(precision, recall):
    numerator = precision * recall 
    denominator = precision + recall 
    return 2 * (numerator / denominator)

f1_score = calculate_f1(precision, recall)

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





