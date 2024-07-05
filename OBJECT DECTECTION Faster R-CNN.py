import torchvision
from torchvision import transforms
import torch
from torch import no_grad
import requests
import  cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import warnings
warnings.filterwarnings("ignore")
def get_predictions(pred,threshold=0.8,objects=None):
    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(box[0], box[1]), (box[2], box[3])]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes


def draw_box(predicted_classes, image, rect_th=10, text_size=3, text_th=3):
    img = (np.clip (
        cv2.cvtColor ( np.clip ( image.numpy ( ).transpose ( (1 , 2 , 0) ) , 0 , 1 ) , cv2.COLOR_RGB2BGR ) , 0 ,
        1 ) * 255).astype ( np.uint8 ).copy ( )
    for predicted_class in predicted_classes :
        label = predicted_class [ 0 ]
        probability = predicted_class [ 1 ]
        box = predicted_class [ 2 ]
        pt1 = (int ( box [ 0 ] [ 0 ] ) , int ( box [ 0 ] [ 1 ] ))
        pt2 = (int ( box [ 1 ] [ 0 ] ) , int ( box [ 1 ] [ 1 ] ))
        cv2.rectangle ( img , pt1 , pt2 , (0 , 255 , 0) , rect_th )
        cv2.putText ( img , label , pt1 , cv2.FONT_HERSHEY_SIMPLEX , text_size , (0 , 255 , 0) , thickness = text_th )
        cv2.putText ( img , label + ": " + str ( round ( probability , 2 ) ) , pt1 , cv2.FONT_HERSHEY_SIMPLEX ,
                      text_size , (0 , 255 , 0) , thickness = text_th )
    plt.imshow ( cv2.cvtColor ( img , cv2.COLOR_BGR2RGB ) )
    plt.show ( )
    del img
    del image
def save_RAM(image_=False):
    global image,img,pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)
model_=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()
for name,param in model_.named_parameters():
    param.requires_grad=False
def model(x):
    with torch.no_grad():
        yhat=model_(x)
    return yhat
COCO_INSTANCE_CATEGORY_NAMES=[ '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
transform=transforms.Compose([transforms.ToTensor()])
image_path=str(input("ENTER A IMAGE NAME WITH EXTENSION (eg- photo1.png) :"))
image=Image.open(image_path)
img=transform(image)
pred=model([img])
pred_thresh=get_predictions(pred,threshold = 0.70)
draw_box(pred_thresh,img,rect_th = 1,text_size = 0.5,text_th = 1)
del pred_thresh
save_RAM(image_ = True)
