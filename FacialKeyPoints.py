#Facial Key Points

import pandas as pd
import statistics
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow


#Load facial key points data from data directory
keyfacial_df = pd.read_csv('/Users/Marina/PycharmProjects/AIMasterClass/EmotionAI/Data/data.csv')
keyfacial_df.info()
print(keyfacial_df.isnull().sum())
print(keyfacial_df['Image'].shape)
keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96,96))
print(keyfacial_df['Image'][0].shape)

#Mini Challenge #1
print(keyfacial_df.describe())
print(min(keyfacial_df['right_eye_center_x']))
print(max(keyfacial_df['right_eye_center_x']))
print(statistics.mean(keyfacial_df['right_eye_center_x']))


