import tempfile

import streamlit as st
import matplotlib.pyplot as plt

from src.predict import denoise_image

st.set_page_config(
    page_title="Emergency Vehicle Image Denoising",
    layout="wide"
    )

st.title("Emergency Vehicle Image Denoising")
st.write(
    "Upload a noisy emergency vehicle image and "
    "the Autoencoder will generate a denoised version."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    noisy_image, denoised_image = denoise_image(
        image_path
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Noisy Image")

        fig, ax = plt.subplots()

        ax.imshow(noisy_image)
        ax.axis("off")
  
        st.pyplot(fig)

    with col2:

        st.subheader("Denoised Image")

        fig, ax = plt.subplots()

        ax.imshow(denoised_image)
        ax.axis("off")

        st.pyplot(fig)
