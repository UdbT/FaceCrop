import os
import json

import requests
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/detect_largest_face'

# Prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

image_path = './sample/0.JPG'
img = cv2.imread(image_path)

# Encode image as jpeg
_, img_encoded = cv2.imencode(os.path.splitext(image_path)[1], img)

# Send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
if response.status_code != 200:
    print(response.text)
    exit()

# Decode response
bound = json.loads(response.text)

x = bound['left']
y = bound['top']
w = bound['right'] - x
h = bound['buttom'] - y

# Draw box over face
cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# Display output image
cv2.imshow("face detection with dlib", img)
cv2.waitKey(3000)