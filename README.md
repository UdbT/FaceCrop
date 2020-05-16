# FaceCrop package
To crop and return face from the given image.
```
    import numpy as np
    import cv2

    img_path = <IMAGE_PATH>

    # Detect largest face
    lg_face, bound = detect_largest_face(image_path=img_path)

    # Read image
    image = cv2.imread(img_path)

    x = bound['left']
    y = bound['top']
    w = bound['right'] - x
    h = bound['buttom'] - y

    # Draw box over face
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display output image
    im1 = cv2.resize(image,(400,300))
    im2 = cv2.resize(lg_face,(400,300))
    imstack = np.concatenate((im1, im2), axis=1)

    cv2.imshow("Largest face", imstack)
    cv2.waitKey(5000)
```
# API
Provide API for user to get bounding box of the largest face in the given image.