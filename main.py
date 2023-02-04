from fastapi import FastAPI, Request, File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import cv2
import glob
import os
import uuid
from pathlib import Path

app = FastAPI(title='HTML Example')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
print('Hello from Main')
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/html", StaticFiles(directory="html"), name="html")
#templates = Jinja2Templates(directory="HTML")
#BASE_DIR = Path(__file__).resolve().parent
print(os.listdir())
templates = Jinja2Templates(directory="html")

@app.get('/home', response_class=HTMLResponse)
async def home(request:Request):    
    return templates.TemplateResponse("home.html",{"request": request})

@app.get('/index', response_class=HTMLResponse)
async def index(request:Request):    
    return templates.TemplateResponse("index.html",{"request": request})

@app.get('/team', response_class=HTMLResponse)
async def index(request:Request):    
    return templates.TemplateResponse("team.html",{"request": request})

@app.get('/project', response_class=HTMLResponse)
async def index(request:Request):    
    return templates.TemplateResponse("project.html",{"request": request})

@app.get('/yolov5_t1', response_class=HTMLResponse)
async def index(request:Request):    
    return templates.TemplateResponse("yolov5_t1.html",{"request": request})

@app.get('/yolov5_t2', response_class=HTMLResponse)
async def index(request:Request):    
    return templates.TemplateResponse("yolov5_t2.html",{"request": request})

@app.get('/index/{id}', response_class=HTMLResponse)
async def index(request:Request, id:str):    
    return templates.TemplateResponse("home.html",{"request": request, "id": id})

@app.get('/image/{id}', response_class=FileResponse)
async def image(request:Request, id:str):    
    return FileResponse("./tmp/"+id)

@app.get('/html/{id}', response_class=HTMLResponse)
async def image(request:Request, id:str):    
    return FileResponse("./html/"+id)


#YOLOv5
yolov5Model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#yolov5Model = torch.hub.load('ultralytics/yolov5', 'custom', 'Soumya_yolov5.pt')
@app.post("/yolov5_a1",tags=['YOLOv5 Model'])
def predict_imageYOLOv5(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = "jpg"
        file_location = f"./tmp/{file_id}.{file_extension}"
        result_file_location = f"./tmp/{file_id}_res.{file_extension}"
        srcFileName = f"{file_id}.{file_extension}"
        resFileName = f"{file_id}_res.{file_extension}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        imgs = []
        imgs.append(file_location)
        results = yolov5Model(imgs)
        r_img = results.render()
        #results.save()
        im_rgb = cv2.cvtColor(r_img[0], cv2.COLOR_BGR2RGB) # Because of OpenCV reading images as BGR
        cv2.imwrite(result_file_location, im_rgb)
        #cv2_imshow(im_rgb)
        #return {"info": f"file '{file.filename}' saved at '{file_location}'"}
        #return {"info": f"file '{file.filename}' result saved at '{result_file_location}'"}
        return {"info": f"file '{file.filename}' result saved at '{result_file_location}'",
                "req_file": f"{srcFileName}",
                "res_file": f"{resFileName}"
                }
    except Exception as e:
        return {"error": f"{e}"}
    
@app.post("/yolov5_a2",tags=['YOLOv5 Model'])
def predict_vedioYOLOv5(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = "mp4"
        file_location = f"./tmp/{file_id}.{file_extension}"
        result_file_location = f"./tmp/{file_id}_res.{file_extension}"
        srcFileName = f"{file_id}.{file_extension}"
        resFileName = f"{file_id}_res.{file_extension}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        video = cv2.VideoCapture(file_location)
        ok,frame=video.read()
        # We need to set resolutions.
        # so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        #size = (frame_width/10, frame_height/10)
        #size =  (640,480)

        # Below VideoWriter object will create
        # a frame of above defined The output 
        # is stored in 'filename.avi' file.
        #cv2.VideoWriter_fourcc(*'MJPG')
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #fourcc = cv2.VideoWriter_fourcc(*'X264')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        resultVW = cv2.VideoWriter(result_file_location, 
                                fourcc,
                                30, size)
        
        while True:
            ok,frame=video.read()
            if not ok:
                    break
            #ok,bbox=tracker.update(frame)            
            if ok:
                results = yolov5Model(frame)
                r_img = results.render()
                im_rgb = cv2.cvtColor(r_img[0], cv2.COLOR_BGR2RGB)
                resultVW.write(im_rgb)
                    #(x,y,w,h)=[int(v) for v in bbox]
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
            else:
                cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            #cv2.imshow('Tracking',frame)
                    # Write the frame into the
                    # file 'filename.avi'
            resultVW.write(frame)
            if cv2.waitKey(1) & 0XFF==27:
                    break
        #imgs = []
        #imgs.append(file_location)
        #results = yolov5Model(imgs)
        #r_img = results.render()
        #results.save()
        #im_rgb = cv2.cvtColor(r_img[0], cv2.COLOR_BGR2RGB) # Because of OpenCV reading images as BGR
        #cv2.imwrite(result_file_location, im_rgb)

        # When everything done, release 
        # the video capture and video 
        # write objects
        video.release()
        resultVW.release()
            
        # Closes all the frames
        cv2.destroyAllWindows()

        #cv2_imshow(im_rgb)
        #return {"info": f"file '{file.filename}' saved at '{file_location}'"}
        #return {"info": f"file '{file.filename}' result saved at '{result_file_location}'"}
        return {"info": f"file '{file.filename}' result saved at '{result_file_location}'",
                "req_file": f"{srcFileName}",
                "res_file": f"{resFileName}"
                }
    except Exception as e:
        return {"error": f"{e}"}