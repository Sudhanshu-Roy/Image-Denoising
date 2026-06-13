from src.model import build_autoencoder
from src.preprocess import preprocess_image

model = build_autoencoder()

model.load_weights(
    "models/denoising_autoencoder.weights.h5"
)

def denoise_image(image_path):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image.reshape(1,224,224,3),
        verbose=0
    )

    denoised_image = prediction[0]

    return image, denoised_image
