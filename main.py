import cv2
import pathlib
import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras import layers

# директория обучения
train_dir = pathlib.Path("./datasets/fruits/train/")
# директория тестирования
test_dir = pathlib.Path("./datasets/fruits/test/")
# директория валидации
val_dir = pathlib.Path("./datasets/fruits/validation/")
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir)
test_ds = tf.keras.utils.image_dataset_from_directory(test_dir)
val_ds = tf.keras.utils.image_dataset_from_directory(val_dir)
class_names = dict(zip(train_ds.class_names, range(len(train_ds.class_names))))
num_classes = len(class_names)

train_generator = ImageDataGenerator(
preprocessing_function = mobilenet_v2.preprocess_input,
rotation_range = 32,
zoom_range = 0.2,
width_shift_range = 0.2,
height_shift_range = 0.2,
shear_range = 0.2,
horizontal_flip = True,
fill_mode = "nearest")

train = train_generator.flow_from_directory(train_dir,
target_size = (img_height,img_width),
# изображение имеет 3 цветовых канала
color_mode = "rgb",
# создаем бинарные признаки меток класса 
class_mode = "categorical",
batch_size = 32,
shuffle = True,
seed = 123)

validation = train_generator.flow_from_directory(val_dir,
target_size = (img_height,img_width),
# изображение имеет 3 цветовых канала
color_mode = "rgb",
# создаем бинарные признаки меток класса 
class_mode = "categorical",
batch_size = 32,
shuffle = True,
seed = 123)

mobilenet_ = MobileNetV2(
input_shape = (img_height,img_width,3),
include_top = False,
weights = 'imagenet',
pooling = 'avg')

mobilenet_.trainable = False

inputs = mobilenet_.input
x = Dense(128, activation = 'relu')(mobilenet_.output)
x = Dense(128, activation = 'relu')(x)
outputs = Dense(num_classes , activation = 'softmax')(x)

mobilenet = Model(inputs = inputs, outputs = outputs)

early_stopping = EarlyStopping(
	monitor='val_loss',
	mode='min',
	patience = 2,
	verbose=1,
	restore_best_weights=True,
)
checkpoint = ModelCheckpoint('./fruit224mobile.h5',
                        	monitor = 'val_loss',
                        	mode = 'min',
                       	save_best_only = True)

callbacks = [early_stopping, checkpoint]

mobilenet.compile(optimizer='adam', loss ='categorical_crossentropy',metrics = ['accuracy'])

history = mobilenet.fit(
train, validation_data = validation,
batch_size = 32,
epochs = 20,
callbacks = callbacks)

test = train_generator.flow_from_directory(test_dir,
target_size = (224,224),
color_mode = "rgb",
class_mode = "categorical",
batch_size = 32,
shuffle = False)

(eval_loss, eval_accuracy) = mobilenet.evaluate(test)

# получаем предсказанные значения от тестовых изображений
pred = mobilenet.predict(test)
# получаем номер класса с максимальным весом
pred = np.argmax(pred,axis=1)
# сопоставляем классы
labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
# получаем предсказанные классы
y_test = [labels[k] for k in test.classes]
# оцениваем точность
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, pred)
print(f'Accuracy on the test set: {100*acc:.2f}%')