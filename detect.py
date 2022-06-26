import os
import argparse
import  cv2
import glob
import time
import PIL
import detectron2
import numpy as np

from detectron2.utils.logger import setup_logger
from PIL import Image
from torch import pixel_shuffle
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor 
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures.instances import Instances
from densepose.vis.extractor import BoundingBoxExtractor


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser( )
#parser.add_argument('--path', type=dir_path)

parser.add_argument('--input_path', type=dir_path)
parser.add_argument('--extend_path', type=dir_path)
parser.add_argument('--input_format', type= str)
parser.add_argument('--output_format', type=str)

args = parser.parse_args()

def aspectratio(extend_image):
    extend_height,extend_width,ed=extend_image.shape
    extend_ratio=extend_width/extend_height
    print("Extend Image Ratio",extend_ratio)


def predictor_coordinates():
    
    
    for file in glob.glob( args.input_path + '/*.'+args.input_format):
        
        img= cv2.imread(file) 
        x= file.rsplit(".",1)[0]
        z= len(args.input_path)+1
        y=(x[z:])
        imgarr = np.array(img)


        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        outputs = predictor(imgarr)
        prediction_class=outputs["instances"].pred_classes
       
        
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(imgarr[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    

        #convert array to image
        pimg = Image.fromarray(out.get_image())

        #extract bounding box from Box class
        x=BoundingBoxExtractor()
        xy=x(outputs["instances"])
        

        #prediction_class tensor to numpy
        pa = prediction_class.detach().cpu().data.numpy()
        pc=pa.astype(int)
       
     
        #tensor to numpy
        na = xy.detach().cpu().data.numpy()
        nb=na.astype(int)
        object=[]
       
        #checking index in pred_class
        for index, item in enumerate(pc, start=0):   
            if item==0:
                object=nb[index]
                
                    
        #return x,y,width,height   
        xx,yy,ww,hh = object
        
        

        #crop the detected object
        ll =imgarr[int(yy):int(yy+hh), int(xx):int(xx+ww)]
        crop_img= Image.fromarray(ll)
        crop_ratio=ww/hh
            
        #adding pad
        b,g,r =ll[1, 1] #Pixel Color
        blue=int(b)
        green=int(g)
        red=int(r)
        
    
        if crop_ratio<0.75:
            new_width = (hh * 0.75)
            required_width=int((new_width - ww) / 2)
            # print('Required Width',required_width)
            extend_image = cv2.copyMakeBorder(ll,0,0,required_width,required_width,cv2.BORDER_CONSTANT,value=(blue,green,red,0))
            aspectratio(extend_image)
            cv2.imwrite ( args.extend_path + '/'+ y + '.' + args.output_format,extend_image)
        else:
            new_height=(ww/ 0.75)
            required_height= int((new_height- hh) / 2)
            # print('Required Height', required_height)
            extend_image = cv2.copyMakeBorder(ll,required_height,required_height,0,0,cv2.BORDER_CONSTANT,value=(blue,green,red,0))
            aspectratio(extend_image)
            cv2.imwrite ( args.extend_path + '/'+ y + '.' + args.output_format,extend_image)
       

       
    return xx,yy,ww,hh,crop_img,crop_ratio    


  
predictor_coordinates()



    

  
