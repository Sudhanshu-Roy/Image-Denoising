import cv2
import numpy as np

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    return image
