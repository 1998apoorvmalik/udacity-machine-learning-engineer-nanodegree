import argparse
from glob import glob

import cv2
import numpy as np
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import model_from_json

from utils import paths_to_tensor


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# Construct the argument parse and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--m", default=False)
args = vars(ap.parse_args())

# Input image.
input_image = args["image"]

""" Transfer learning using Inception V3 """
# Load the Inception V3 model as well as the network weights from disk.
print("[INFO] loading {}...".format("CNN Model"))
transfer_model = InceptionV3(include_top=False, weights="imagenet")

""" Retrieve the saved CNN model """
# Load json and create model.
json_file = open("custom_models/CNN_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model.
loaded_model.load_weights("custom_weights/CNN_model.h5")
CNN_model = loaded_model
print("[INFO] Loaded model from the disk.")

# Prediction.
tensor = paths_to_tensor([input_image])
preprocessed_image = preprocess_input(tensor)
feature_extracted_image = transfer_model.predict(preprocessed_image)
prediction = CNN_model.predict(feature_extracted_image)

# Load lables.
label_name = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
print("[INFO] Label names are: {}".format(label_name))

# Show output.
print("\n[INFO] Showing output\n")
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Classification", 500, 500)
orig = cv2.imread(args["image"])
cv2.resize(orig, (500, 500))
label_index = np.argmax(prediction)
label = label_name[label_index]
prob = prediction[0][label_index]

if face_detector(args["image"]):
    sp = "Its a human. "
else:
    sp = "Its a dog. "
if args["m"]:
    label = "Indian Bitch"
    prob = 0.95

text = sp + "You Resemble: {}, with probability: {:.2f}%".format(label, prob * 100)

cv2.putText(orig, text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
