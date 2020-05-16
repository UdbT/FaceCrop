
from datetime import datetime
import os

from flask import Flask, request, Response, abort
import jsonpickle
import numpy as np
import cv2

import facecrop

app = Flask(__name__)

@app.route('/detect_largest_face', methods=['POST'])
def detect_largest_face():
    '''
    Returns the bounding box of the largest face in given image

    '''
    r = request

    # Convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)

    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save image to tmp directory
    img_tmp_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'tmp', 'TMP_'+datetime.now().strftime("%d%m%Y_%H%M%S")+'.jpg')
    cv2.imwrite(img_tmp_path, img)

    # Find largest face in image
    try:
        lg_face, bound = facecrop.detect_largest_face(img_tmp_path)
    except Exception as err:
        return Response(str(err), status=500)

    # Build a response dict
    response = bound

    # Encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
