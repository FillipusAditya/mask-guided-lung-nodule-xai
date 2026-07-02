import numpy as np
import pydicom
from PIL import Image
import os

def dicom_to_png(dicom_path, output_path):
    """
    Convert a DICOM (.dcm) image file to a PNG image.
    """

    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)

    # Extract pixel data and convert to float for processing
    pixel_array = dicom_data.pixel_array.astype(float)

    # Normalize pixel values to range [0, 255]
    # - Ensure no negative values using np.maximum
    # - Divide by max value to scale between 0 and 1
    # - Multiply by 255 to convert to image intensity range
    normalized_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0

    # Convert to unsigned 8-bit integer (required for image saving)
    image_uint8 = normalized_image.astype(np.uint8)

    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_uint8)

    # Save the image as PNG
    image.save(output_path)

    print(f"Conversion complete: {output_path}")

def display_dicom_metadata(dicom_path):
    """
    Display a DICOM metada on terminal
    """
    
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Display the DICOM metadata
    print(dicom_data)
    

# Usage
dcm_img_path = "./images/sample.dcm"
output_path = "./images/sample.png"

# dicom_to_png(dcm_img_path, output_path)
display_dicom_metadata(dcm_img_path)