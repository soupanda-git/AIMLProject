from fastapi import FastAPI,File,UploadFile
from tensorflow.keras import models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from io import BytesIO
from PIL import Image
app = FastAPI()
model = ResNet50(weights = 'imagenet')

@app.post("/upload",tags=['Resnet50 Pretrained Model'])
def predict_image(file:bytes=File(...)):
    pil_image = Image.open(BytesIO(file))
    img = np.asarray(pil_image.resize((224,224)))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    result = decode_predictions(model.predict(x),3)[0]
    response = []
    for i,res in enumerate(result):
        resp = {}
        resp['class'] = res[1]
        resp['responsibility'] = res[0]
        response.append(resp)
    return response