import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
import cv2

X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("labels.csv")["labels"]

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train,  y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
x_train_scale = x_train / 255.0
x_test_scale = x_test / 255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scale, y_train)

def get_prediction(image):
  im_pil = Image.open(image)
  image_bw = im_pil.convert('L')
  image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
  pixel_filter = 20
  min_pixel = np.percentile(image_bw_resized, pixel_filter)
  image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
  max_pixel = np.max(image_bw_resized)
  image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
  test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
  test_pred = clf.predict(test_sample)
  return test_pred[0]


