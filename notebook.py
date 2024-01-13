#IMPORTS
import os
import cv2
import tqdm
import pickle
import numpy as np
import csv
import sys
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#PATH
dataset_path = "DATASET"

#MEDIAPOSE
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
    min_detection_confidence=0.5)

#CLASS LABELS
class_labels = {
    "downdog": 0,
    "goddess": 1,
    "plank": 2, 
    "tree": 3,
    "warrior2": 4
}
pose_label = ["Downdog","Goddess","Plank","Tree","Warrior II"]

#MODEL
def create_model():
    model = keras.Sequential()
    model.add(
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation=keras.activations.relu,
            padding="same",
            input_shape=(33, 2)
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling1D())
    model.add(
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation=keras.activations.relu,
            padding="same",
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation=keras.activations.softmax))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    return model

#TRAIN MODEL
def train_model(model, train_dataset, val_dataset):
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
    model.save("save/predict.model")
    return history

#PLOTTING ACCURACY
def plot_accgraph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

#LOAD DATASET
def load_dataset(
        dataset_label,
        data_path,
        class_labels,
    ):
    data = []
    labels = []
    for class_name in class_labels.keys():
        path = f"{data_path}/{dataset_label}/{class_name}"
        print(f"Loading {dataset_label} Data for Class Name: {class_name}")
        for filename in tqdm.tqdm(os.listdir(path), position=0):
            image = cv2.imread(f"{path}/{filename}")
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue
            sample = []
            for lm in results.pose_landmarks.landmark:
                sample.append((lm.x, lm.y))
            data.append(sample)
            label_sample = np.zeros(5)
            label_sample[class_labels[class_name]] = 1
            labels.append(label_sample)
    return np.array(data), np.array(labels)

#LOAD DATA
def load_data(data_path, class_labels):
    train_data, train_labels = load_dataset(
        "TRAIN",
        data_path,
        class_labels,
    )
    test_data, test_labels = load_dataset(
        "TEST",
        data_path,
        class_labels,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    return train_dataset, test_dataset

#LOADING DATASET
train_dataset, test_dataset = load_data(dataset_path, class_labels)

#LOADING MODEL
model = create_model()

#TRAINING MODEL
history=train_model(model, train_dataset, test_dataset)

#ACCURACY
plot_accgraph(history)

#TRAINED MODEL
model = keras.models.load_model("save/predict.model")

#MAKE PREDICTIONS
def predict_with_static_image(model,filepath):
    input_frame = cv2.imread(filepath)
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image=input_frame)
    output_image = input_frame.copy()
    label = "Unknown Pose"
    accuracy = 0
    mp_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    sample = []
    for lm in results.pose_landmarks.landmark:
        sample.append((lm.x, lm.y))
    prediction = model(np.array(sample)[np.newaxis, :, :])
    p=np.array(prediction)
    label=pose_label[np.argmax(p)]
    accuracy = str(round(p[0][np.argmax(p)]*100,3)) 
    if(float(accuracy)<50):
        label = "Unknown Pose"
    return label,accuracy,output_image

#OUTPUT
def output(lab,acc,out):
    plt.figure(figsize = [10, 10])
    if(lab=="Unknown Pose"):
        plt.title(lab)
    else:
        plt.title(f"Pose: {lab} | Accuracy: {acc}%")
    plt.axis('off')
    plt.imshow(out)
    plt.show()

#PATH
path="Testing_Images/untitled-1018jpg.jpg"

#MAKING PREDICTIONS
lab,acc,out=predict_with_static_image(model,path)
output(lab,acc,out)

# Machine Learning

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
    min_detection_confidence=0.5)

#PLOTTING SAMPLE IMAGE
image = cv2.imread('sample.png')
plt.figure(figsize = [10, 10])
plt.title("Sample Image")
plt.axis('off')
plt.imshow(image[:,:,::-1])
plt.show()

results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
img_copy = image.copy()
if results.pose_landmarks:
    mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
    plt.figure(figsize = [10, 10])
    plt.title("Output Image")
    plt.axis('off')
    plt.imshow(img_copy[:,:,::-1])
    plt.show()

results.pose_landmarks

num_coords=len(results.pose_landmarks.landmark)
print(num_coords)

landmarks = ['Class']
for val in range(1, num_coords+1):
    landmarks += ['X{}'.format(val), 'Y{}'.format(val), 'Z{}'.format(val)]

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

images_in_folder = 'DATASET/TRAIN'
csv_out_path = 'coords.csv'
class_names = ["Down Dog","Goddess","Plank","Tree","Warrior II"]
folder_names = ["downdog","goddess","plank","tree","warrior2"]

def getcoords(class_name,images_in_folder):    
    image_names = [n for n in os.listdir(images_in_folder) if not n.startswith('.')]
    print('Inserting Data in CSV of class',class_name, file=sys.stderr)
    for image_name in tqdm.tqdm(image_names, position=0):
        input_frame = cv2.imread(os.path.join(images_in_folder,image_name))
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image=input_frame)
        try:
            pose_landmarks = result.pose_landmarks.landmark
            row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks]).flatten())
            row.insert(0, class_name)
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
        except:
            pass

#INSERTING LANDMARKS IN CSV
for i in range(len(class_names)):
    getcoords(class_names[i],os.path.join(images_in_folder,folder_names[i]))

#READING DATA FROM CSV
df = pd.read_csv('coords.csv')
df.shape

#STORING FEATURES AND TARGET VALUE
df_copy=df.copy(deep=True)
X = df_copy.drop('Class', axis=1)
y = df_copy['Class'] 

#SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#MAKING PIPELINES
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=5000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    'kn':make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

#FITTING MODEL
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

#MODELS
fit_models

#EVALUATING MODELS
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo,':',accuracy_score(y_test, yhat))

#STORING BEST MODEL
with open('detect_pose.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)

#LOADING MODELc
with open('detect_pose.pkl', 'rb') as f:
    model = pickle.load(f)

#FUNCTION TO MAKE DETECTION USING IMAGE
def make_detections(image):
    input_frame = cv2.imread(image)
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image=input_frame)
    output_image = input_frame.copy()
    label = "Unknown Pose"
    accuracy = 0
    mp_drawing.draw_landmarks(image=output_image, landmark_list=result.pose_landmarks, 
                                  connections=mp_pose.POSE_CONNECTIONS)
    pose_landmarks = result.pose_landmarks.landmark
    row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks]).flatten())
    X = pd.DataFrame([row])
    label = model.predict(X)[0]
    body_language_prob = model.predict_proba(X)[0]
    accuracy = str(round(body_language_prob[np.argmax(body_language_prob)]*100,3))
    if(float(accuracy)<50):
        label = "Unknown Pose"
    return label,accuracy,output_image

#OUTPUT
def output(lab,acc,out):
    plt.figure(figsize = [10, 10])
    if(lab=="Unknown Pose"):
        plt.title(lab)
    else:
        plt.title(f"Pose: {lab} | Accuracy: {acc}%")
    plt.axis('off')
    plt.imshow(out)
    plt.show()

#PREDICTION
lab,acc,out=make_detections('sample.png')
output(lab,acc,out)



