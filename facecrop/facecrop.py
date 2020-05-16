"""
    Return largest face from the given image
"""

# import required packages
import os

import cv2
import dlib
import numpy

def load_image(image_path:str) -> numpy.ndarray:
    '''
    Return loaded image from the specified file.
    
    Parameters
    ----------
    image_path: string
    
    Raises
    ----------
    ValueError
        - Could not load image
        - Image dimension exceeded limit

    Returns
    ----------
    numpy.ndarray (3d array of loaded image)
    '''

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read input image \"{}\"".format(image_path))
    
    img_height, img_width = image.shape[:2]
    if img_height > 1024 or img_width > 1024:
        raise ValueError("Image dimension exceeded the limit of (1024x1024)")
    
    return image
    
def detect_face(image:numpy.ndarray) -> dlib.rectangles:
    '''
    Return face detected bounds
    
    Parameters
    ----------
    image: numpy.ndarray (ndarray returned from cv2.imread())
    
    Raises
    ----------
    ValueError
        - No detected face in the image

    Returns
    ----------
    dlib.rectangles (bound of detected faces)
    '''
    # Initialize hog based face detector
    hog_face_detector = dlib.get_frontal_face_detector()

    # Apply face detection (hog)
    faces_hog = hog_face_detector(image, 1)
    if not faces_hog:
        raise ValueError("No detected face in this image!")

    return faces_hog

def get_largest_face(image:numpy.ndarray, faces:dlib.rectangles) -> numpy.ndarray:
    '''
    Return largest cropped face from given bounds
    
    Parameters
    ----------
    image: numpy.ndarray
    faces: dlib.rectangles
    
    Returns
    ----------
    numpy.ndarray (largest cropped face)
    '''
    face_img = None

    # Loop over detected faces
    for face in faces:
        
        # Cropping
        sub_face = image[face.top():face.bottom(), face.left():face.right()]

        if not (face_img is None):
            if sub_face.size > face_img.size:
                face_img = sub_face
        else:
            face_img = sub_face
    
    return face_img


def detect_largest_face(image_path:str, output_dir:str='') -> numpy.ndarray:
    '''
    Return and save largest cropped face from given image to specified directory

    Parameters
    ----------
    image_path: string
    output_dir: string
        By default, the image will not be saved
    
    Returns
    ----------
    numpy.ndarray (largest cropped face)
    '''

    image = load_image(image_path)
    largest_face = get_largest_face(image, detect_face(image))

    # Save image to specified directory
    if output_dir:
        # Create directory if not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, os.path.split(image_path)[1]), largest_face)

    return largest_face

if __name__ == '__main__':

    img_path = './sample/0.jpeg'

    # Display output image
    cv2.imshow("Face", detect_largest_face(image_path=img_path, output_dir='./result'))
    cv2.waitKey(3000)