import argparse
import time
from sys import platform

from models.models_yolo import *
from datasetutils.datasetsv2 import *
from utils.utils import *


import numpy as np
from random import sample
import sys, os
import time
import cv2
import imutils
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
import utils.utils_crnn as utils_crnn
import models.crnn as crnn
import alphabets
import params

from torch.utils.data import DataLoader
from datasetutils.dataset_v2_new import baiduDataset

from torch.utils import data
from models.gan import discriminator,generator
from datasetutils.datasets_wgan import VOCDataSet
from torch.optim import Adam
from utils.loss import BCE_Loss
from utils.transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor

import torchvision.transforms as transforms
import tqdm
from utils.Criterion import Criterion
from PIL import Image

from orderutils.crack_pro import *



str1 = alphabets.alphabet


alphabet = str1
nclass = len(alphabet)+1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def draw_cross_and_order_on_Image(image, centers):
    img=image.copy()
    

    nimage=img.transpose((1,2,0))
    #cv2.imwrite("OrderImage.jpg",nimage*255)
    
    width=image.shape[2]
    height=image.shape[1]
    #print(image.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    drawimg=(nimage*255).copy()
    
    for i in range(len(centers)):
        drawx=int(centers[i][0]*width)
        drawy=int(centers[i][1]*height)
        print(drawx,drawy)
        
        img = cv2.putText(drawimg, str(i+1), (drawx,drawy), font, 1.2, (0, 0, 255), 3)
        cv2.circle(drawimg,(drawx,drawy),5,(255,0,0),-1)
        
    return drawimg    
        
        

def preprocessing(image):
   ## already have been computed
    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(params.mean).div_(params.std)   
    
    return image  
def image_Filter(image,thresholdvalue):
    
    
    ret, thresh = cv2.threshold(image, thresholdvalue, 255, cv2.THRESH_BINARY)
    thresh=np.array(thresh,dtype=np.uint8)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    frame=thresh.copy()
    for i in range(len(contours)):
        area= cv2.contourArea(contours[i])
        if area <5 :
            #print("area",area)
            cv2.drawContours(frame,contours[i],-1,0,1)
    
    return frame
    
def prepare_crnn_model(weights_init,params_nh,params_crnn):  
    nclass= len(params.alphabet) + 1
    nc = 1
    # cnn and rnn
    crnnnet = crnn.CRNN(32, nc, nclass, params_nh)

    crnnnet.apply(weights_init)
    if params.crnn != '':
        print('loading pretrained model from %s' % params_crnn)
        crnnnet.load_state_dict(torch.load(params_crnn))
    return crnnnet

def crnn_recognition(image, model,iindex,indexnumber):
    
    if torch.cuda.is_available():
        model = model.cuda()

    converter = utils_crnn.strLabelConverter(alphabet)

     
    timage=preprocessing(image)  
    #timage = timage.to(device)    

    if torch.cuda.is_available():
        timage = timage.cuda()
    timage = timage.view(1, *timage.size())
    timage = Variable(timage)

    model.eval()
    preds = model(timage)    
    softmax_adv = F.softmax(preds, dim=2)

    softmaxsort_adv = torch.argsort(softmax_adv)
    preds = torch.argmax(preds, 2)

    
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    
    
    printadv=softmaxsort_adv[len(softmaxsort_adv)-1]
    #print(printadv[:,printadv.shape[1]-10:printadv.shape[1]])
    preds = printadv[:,printadv.shape[1]-indexnumber:printadv.shape[1]].transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred

def crnn_recognition_mono(image, model,iindex,indexnumber):
    if torch.cuda.is_available():
        model = model.cuda()

    converter = utils_crnn.strLabelConverter(alphabet)


    timage=torch.from_numpy(image).unsqueeze(0)
    

    timage.sub_(params.mean).div_(params.std)
    if torch.cuda.is_available():
        timage = timage.cuda()
    timage = timage.view(1, *timage.size())
    timage = Variable(timage)


    model.eval()
    preds = model(timage)
 
    
    softmax_adv = F.softmax(preds, dim=2)
    softmax_advtt = F.softmax(preds, dim=2)

    softmaxsort_adv = torch.argsort(softmax_adv)
    preds = torch.argmax(preds, 2)

    
    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    printadv=softmaxsort_adv[len(softmaxsort_adv)-1]

    #print(printadv[:,printadv.shape[1]-10:printadv.shape[1]])
    preds = printadv[:,printadv.shape[1]-indexnumber:printadv.shape[1]].transpose(1, 0).contiguous().view(-1)
    


    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('results: {0}'.format(sim_pred))
    return sim_pred


def prepareleftbottom(copyimg,half):
    leftbottomimageorigin=copyimg[:,params.color_width:params.color_height,0:params.color_width].copy()
  
    leftbottomimage=leftbottomimageorigin.transpose((1,2,0))   
      
    imgray = cv2.cvtColor(leftbottomimage*255,cv2.COLOR_BGR2GRAY)
 
    ret,thresh = cv2.threshold(imgray,240,255,cv2.THRESH_BINARY)  

    thresh=np.array(thresh,np.uint8)
    
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    contour = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)


    nimage=leftbottomimageorigin[:,0:h,0:w].copy()      
    nimage= nimage.transpose((1,2,0))       
        
    sizefordetect=(192,96)
        
    lbimage, *_ = letterbox(nimage*255, sizefordetect[0])
    # Normalize RGB
    lbimage = lbimage[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    lbimage = np.ascontiguousarray(lbimage, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    lbimage /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return lbimage


def get_hanzi_coords(cord,iindex,add,copyimg):
 
    minx=int(cord[0])-add
    maxx=int(cord[2])+add
    miny=int(cord[1])-add
    maxy=int(cord[3])+add
            
    minx_origin=int(cord[0])
    maxx_origin=int(cord[2])
    miny_origin=int(cord[1])
    maxy_origin=int(cord[3])
    widthcharacter=maxx-minx
    heightcharacter=maxy-miny                             
            
    hanzi=(str(iindex+1), 0.5, (((maxx_origin+minx_origin)/2)/copyimg.shape[3],
         ((maxy_origin+miny_origin)/2)/copyimg.shape[2], widthcharacter/copyimg.shape[3],                                                                 heightcharacter/copyimg.shape[2] )) 
    
    center=(((maxx_origin+minx_origin)/2)/copyimg.shape[3],
         ((maxy_origin+miny_origin)/2)/copyimg.shape[2], widthcharacter/copyimg.shape[3],                                                                 heightcharacter/copyimg.shape[2] ) 
    
    image=copyimg[:,:,miny:maxy,minx:maxx]

    if image.shape[2]==0 or image.shape[3]==0:
        return None,None,None

                 
    imgre=torch.squeeze(image,0)
    nimage=imgre.data.cpu().numpy()
    nimage=nimage.transpose((1,2,0))
    
    return nimage,hanzi,center
    
    

def get_rec_word(det,copyimg,model_crnn):
    all_hanzi_lists=[]
    hanzi_list=[]
    preds_list=[]
    if det is not None:
        print(len(det),"character number")

    if det is not None and len(det) > 0:
        
        iindex=0
        add=5    
        for cord in det:
         
            nimage,hanzi,_=get_hanzi_coords(cord,iindex,add,copyimg)
            if nimage is None:
                continue 
            size=(32,32) 
            nimage = cv2.resize(nimage, size, interpolation=cv2.INTER_AREA) 
            nimage = cv2.cvtColor(nimage, cv2.COLOR_BGR2GRAY)
            sim_pred=crnn_recognition_mono(nimage,model_crnn,iindex,5)   
            #print(sim_pred,"hhhh")
            all_hanzi_lists.append(sim_pred)
            hanzi_list.append(hanzi)
            #preds_list()
            preds_list.append(sim_pred[-1])
            iindex=iindex+1
        centers,rec_word=get_order_list(all_hanzi_lists, hanzi_list,preds_list) 
        return centers,rec_word,all_hanzi_lists
    
            
def get_color_rec_word(det,copyimg,G,model_crnn,input_transform):
    all_hanzi_lists=[]
    hanzi_list=[]
    centers=[]
    preds_list=[]
    if det is not None:
        print(len(det),"character number")
        
    if det is not None and len(det) > 0:
        
        iindex=0
        add=8    
        for cord in det:
            nimage,hanzi,center=get_hanzi_coords(cord,iindex,add,copyimg)
            if nimage is None:
                continue             
            size=(64,64) 
            nimage = cv2.resize(nimage, size, interpolation=cv2.INTER_AREA) 
  
            img_cv = np.transpose(nimage, (2, 0, 1))
            img_tensor = torch.from_numpy(img_cv)     
            real_img= input_transform(img_tensor)
            real_img=real_img.unsqueeze(0)  
            real_img = Variable(real_img.cuda(), volatile=True)

            output = G(real_img)   
            
            output = output[0].data.squeeze(0).cpu().numpy()
      
            imagefilter=image_Filter(output*255,250)
            
            
            Image.fromarray((imagefilter).astype(np.uint8)).save("%d%d.jpg" % (iindex, iindex))
            
            sizesmall=(32,32)

            segimg = cv2.resize(imagefilter, sizesmall, interpolation=cv2.INTER_AREA)  
            
            _, thresh = cv2.threshold(255-segimg, 240, 255, cv2.THRESH_BINARY)

            imgcrop = (np.reshape(thresh, (32, 32, 1))).transpose(2, 0, 1)
            
            sim_pred=crnn_recognition(imgcrop,model_crnn,iindex,10)
            
            all_hanzi_lists.append(sim_pred)
            hanzi_list.append(hanzi)
            centers.append(center)
            preds_list.append(sim_pred[-1])            
            iindex=iindex+1
        #centers,rec_word=get_order_list(all_hanzi_lists, hanzi_list)
        _,rec_word=get_order_list(all_hanzi_lists, hanzi_list,preds_list)
        return centers,rec_word,all_hanzi_lists        
                 
def sort_character(rec_word,colorcenters,color_rec_word,color_sim_preds):
    
#     rec_word_number=len(rec_word)
#     color_rec_word_number=len(color_rec_word)
#     if not rec_word_number=color_rec_word_number：
#        return colorcenters,color_rec_word
    orderlist=[]
    counters=0
    findlist=[]
    for i,word in enumerate(rec_word):   
        isfind=-1
        for k in range(len(rec_word)):
            subpreds=color_sim_preds[k]
            for j,pred in enumerate(subpreds):
                #print(pred,word)
                if pred==word:
                    isfind=1
                    counters=counters+1
                    orderlist.append(k)
                    findlist.append(k)
                    break
        if isfind==-1:
             orderlist.append(-1)
        
                
    #print(orderlist,findlist) 
    ordercenters=[]
    
    if counters < len(rec_word):
        fullorderlist=list(range(0,len(rec_word)))
        indexleft=[]
        
        ordercenters=colorcenters.copy()
        for c in range(len(orderlist)):
            if not orderlist[c]==-1:
                ordercenters[c]=colorcenters[orderlist[c]]
                
            else:
                indexleft.append(c)
        #the left ones

        leftlist=list(set(fullorderlist)-set(findlist))
        #print(leftlist,indexleft)

            
        for cl in range(len(indexleft)):
            index=sample(leftlist,1)
            leftlist=list(set(leftlist)-set(index))
            #leftlist.remove(index)
            #print(indexleft[cl],index)
            ordercenters[indexleft[cl]]=colorcenters[index[0]] 
        #print(len(ordercenters),"len(ordercenters)")
        return ordercenters,rec_word
    elif counters==len(rec_word):
        for c in range(len(orderlist)):
            ordercenters.append(colorcenters[orderlist[c]])
        
    return ordercenters,rec_word
            
    
            

def detect(cfg,
           cfg_color,
           data,
           weights,
           weights_color,
           images='data/test',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           save_txt=False,
           save_images=True):
    # Initialize
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)
        
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        colormodel = Darknet(cfg_color, s)
    else:
        colormodel = Darknet(cfg_color, img_size)    
        

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
        
    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        model.half()   
        
        
        
    # Load weights for color
    if weights_color.endswith('.pt'):  # pytorch format
        colormodel.load_state_dict(torch.load(weights_color, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(colormodel, weights_color)    
        


    # Eval mode
    colormodel.to(device).eval()

    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    if opt.half:
        colormodel.half()
        
        
  
        

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # 导入已经训练好的crnn模型    
    model_crnn=prepare_crnn_model(weights_init,params.nh,params.crnn)
    colormodel_crnn=prepare_crnn_model(weights_init,params.nh,params.crnn_color)
 
    
    #生成的分割网络，计算分割的字
    G=torch.nn.DataParallel(generator(n_filters=32)).cuda()
    PATH=params.gan_G  
    G.load_state_dict(torch.load(PATH))
    G.eval()
    
    
    input_transform = Compose([
        #ColorAug(),
        transforms.ToPILImage(),
        ToTensor(),
        Normalize([.5, .5, .5], [.5, .5, .5]),
      ])
    

    # Run inference
    t0 = time.time()
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()  
        
        copyimg=img.copy()
        
        
        lbimage=prepareleftbottom(copyimg,opt.half)
        colorimage=img.copy()
        # Get detections
        leftbottomcopyimg = torch.from_numpy(lbimage).unsqueeze(0).to(device)
        pred, _ = model(leftbottomcopyimg)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        all_hanzi_lists=[]
        hanzi_list=[]
        centers,rec_word,sim_preds=get_rec_word(det,leftbottomcopyimg,model_crnn)
        #print("result",centers,rec_word,sim_preds)
        
        #Get detections from color
        colorcopyimg = torch.from_numpy(colorimage).unsqueeze(0).to(device)
        colorpred, _ = colormodel(colorcopyimg)
        colordet = non_max_suppression(colorpred.float(), conf_thres, nms_thres)[0]
        color_all_hanzi_lists=[]
        color_hanzi_list=[]
        colorcenters,color_rec_word,color_sim_preds=get_color_rec_word(colordet,colorcopyimg,G,colormodel_crnn,input_transform)
        #print("result2",colorcenters,color_rec_word,color_sim_preds)
        if not len(colorcenters)==len(centers):
            continue
        
        showcenters,_=sort_character(rec_word,colorcenters,color_rec_word,color_sim_preds)
        
        #print(len(showcenters))
        
       
        showimage=draw_cross_and_order_on_Image(img,showcenters)
        if not params.saveOrderpath =='':
            cv2.imwrite(str(params.saveOrderpath)+str(i)+".jpg",showimage)
                                  
                                  
    print('Done. (%.3fs)' % (time.time() - t0))
    return showcenters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--cfg_color', type=str, default='cfg/yolov3-spp-color.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/jtrain.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--weights_color', type=str, default='weights/best-Copy_color.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/Samples_color', help='path to images')
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--webcam', action='store_true', help='use webcam')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.cfg_color,
               opt.data,
               opt.weights,
               opt.weights_color,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
