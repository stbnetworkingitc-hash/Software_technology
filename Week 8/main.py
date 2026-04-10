#
# Week 8 Tutorial / Lab Activity
# Insect Image Classification using
# MobileNetV2 and EfficientNetB0
#

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

#
# STEP 0: SET DATASET PATH
#

DATASET_DIR = "insects_dataset/train_data"

IMG_SIZE = (160, 160)
IMG_SIZE_EFF = (224, 224)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 10

print("Dataset path:", DATASET_DIR)
print("Path exists:", os.path.exists(DATASET_DIR))

#
# STEP 1: LOAD DATASET WITH 80/20 SPLIT
#

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes found:", class_names)

#
# STEP 2: SPEED UP DATASET
#

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#
# STEP 3: SHOW SAMPLE IMAGES
#

plt.figure(figsize=(8, 8))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

#
# STEP 4: DATA AUGMENTATION
#

data_augmentation_mobilenet = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

data_augmentation_eff = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

plt.figure(figsize=(8, 8))
for images, labels in train_ds.take(1):
    first_image = images[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_image = data_augmentation_mobilenet(tf.expand_dims(first_image, 0))
        plt.imshow(aug_image[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

#
# FUNCTION TO PLOT TRAINING HISTORY
#

def plot_history(history, model_name):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title(model_name + " Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title(model_name + " Loss")
    plt.legend()

    plt.show()

#
# FUNCTION TO EVALUATE MODEL
#

def evaluate_model(model, dataset, class_names, model_name):
    y_true = []
    y_scores = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.argmax(y_scores, axis=1)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(model_name + " Confusion Matrix")
    plt.show()

    # classification report
    print("\n" + model_name + " Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print("Weighted Precision:", round(precision, 4))
    print("Weighted Recall:", round(recall, 4))
    print("Weighted F1-score:", round(f1, 4))

    # multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_bin.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=class_names[i] + " (AUC = " + str(round(roc_auc, 2)) + ")")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(model_name + " ROC Curve")
    plt.legend()
    plt.show()

#
# STEP 5: MOBILE NET V2 MODEL
#

base_model_mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights=None
)

base_model_mobilenet.trainable = False

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation_mobilenet(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model_mobilenet(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

mobilenet_model = models.Model(inputs, outputs)

mobilenet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

mobilenet_model.summary()

#
# STEP 6: TRAIN MOBILE NET V2
#

history_mobilenet = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

plot_history(history_mobilenet, "MobileNetV2")

val_loss, val_acc = mobilenet_model.evaluate(val_ds)
print("MobileNetV2 Validation Accuracy:", round(val_acc * 100, 2), "%")

evaluate_model(mobilenet_model, val_ds, class_names, "MobileNetV2")

mobilenet_model.save("macroinvertebrates_classifier_mobilenet.h5")
print("MobileNetV2 model saved.")

#
# STEP 7: LOAD DATA AGAIN FOR EFFICIENTNET
#

train_ds_eff = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE_EFF,
    batch_size=BATCH_SIZE
)

val_ds_eff = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE_EFF,
    batch_size=BATCH_SIZE
)

train_ds_eff = train_ds_eff.prefetch(buffer_size=AUTOTUNE)
val_ds_eff = val_ds_eff.prefetch(buffer_size=AUTOTUNE)

#
# STEP 8: EFFICIENTNET B0 MODEL
#

base_model_eff = tf.keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE_EFF + (3,),
    include_top=False,
    weights=None
)

base_model_eff.trainable = False

inputs = layers.Input(shape=IMG_SIZE_EFF + (3,))
x = data_augmentation_eff(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model_eff(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

efficientnet_model = models.Model(inputs, outputs)

efficientnet_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

efficientnet_model.summary()

#
# STEP 9: TRAIN EFFICIENTNET B0
#

history_eff = efficientnet_model.fit(
    train_ds_eff,
    validation_data=val_ds_eff,
    epochs=EPOCHS
)

plot_history(history_eff, "EfficientNetB0")

val_loss, val_acc = efficientnet_model.evaluate(val_ds_eff)
print("EfficientNetB0 Validation Accuracy:", round(val_acc * 100, 2), "%")

evaluate_model(efficientnet_model, val_ds_eff, class_names, "EfficientNetB0")

efficientnet_model.save("macroinvertebrates_classifier_efficientnet.h5")
print("EfficientNetB0 model saved.")