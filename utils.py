# In utils.py
from PIL import Image, ImageOps
import streamlit as st # Keep st for st.error

def load_and_correct_image_orientation(image_source):
    """
    Loads an image from a source (file path or uploaded file object)
    and corrects its orientation based on EXIF data.
    """
    try:
        image = Image.open(image_source)
        # The magic is in exif_transpose
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        # Using st.error here is okay for a simple app, but for true separation,
        # you might log the error and return None, letting the caller handle the UI.
        # For this project, this is fine.
        st.error(f"Could not load or correct image: {e}")
        return None