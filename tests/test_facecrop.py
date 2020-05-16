import sys
import os
import unittest

sys.path.append(".")
from facecrop import facecrop

import numpy
import dlib
import cv2


def compare_image(imageA, imageB):
    from skimage.metrics import structural_similarity as ssim
    return ssim(imageA, imageB, multichannel=True) # Color image

class TestFacecrop(unittest.TestCase):

    def test_load_image(self):
        # Should return image (numpy.ndarray)
        image = facecrop.load_image(image_path='./sample/0.jpeg')
        self.assertIsInstance(image, numpy.ndarray)
    
    def test_load_image_exception(self):
        # Cloud not read exception
        with self.assertRaises(ValueError):
            image = facecrop.load_image(image_path='./sample/99999.jpeg')

        # Image dimension exceeded limit exception
        with self.assertRaises(ValueError):
            image = facecrop.load_image(image_path='./sample/IMG_0102.JPG')
    
    def test_detect_face(self):
        # Should return faces (dlib.rectangles)
        faces = facecrop.detect_face(image=facecrop.load_image(image_path='./sample/0.jpeg'))
        self.assertIsInstance(faces, dlib.rectangles)

    def test_detect_face_exception(self):
        # No face detected exception
        with self.assertRaises(ValueError):
            faces = facecrop.detect_face(image=facecrop.load_image(image_path='./sample/blank.jpeg'))

    def test_get_largest_face(self):
        # Should return largest cropped face (numpy.ndarray)
        image = facecrop.load_image(image_path='./sample/2.jpg')
        lg_face, bound = facecrop.get_largest_face(image=image, faces=facecrop.detect_face(image))
        self.assertIsInstance(lg_face, numpy.ndarray)

        # Compare if it is the same largest face
        mssim = compare_image(lg_face, cv2.imread('./tests/asset/2.jpg'))
        self.assertGreaterEqual(mssim, .9)

    def test_detect_largest_face(self):
        input_image = './sample/0.jpeg'

        # Image should be saved to specified path
        output_path = './tests/result/0.jpeg'
        if os.path.exists(output_path): os.remove(output_path)
        lg_face, bound = facecrop.detect_largest_face(image_path=input_image, output_dir=os.path.split(output_path)[0])
        self.assertTrue(os.path.exists(output_path))

if __name__ == "__main__":
    unittest.main()
