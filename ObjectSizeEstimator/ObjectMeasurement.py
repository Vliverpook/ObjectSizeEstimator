import cv2
import numpy as np
import utils
#默认不使用摄像头
webcam=True
path='1.jpg'
cap=cv2.VideoCapture(0)
#设置亮度
cap.set(10,160)
#设置宽度
cap.set(3,1920)
#设置高度
cap.set(4,1080)
#设置基于a4纸尺寸的放大因子，使得图像展示时不会太小
scale=2
#设置标准a4纸张长宽
wP=210*scale
hP=297*scale


while True:
    if webcam:
        success,img=cap.read()
    else:
        img=cv2.imread(path)
    #调用轮廓函数
    imgContours,conts=utils.getContours(img,minArea=50000,filter=4)
    #获取最大轮廓
    if len(conts) !=0:
        biggest=conts[0][2]
        print(biggest)
        imgWarp=utils.warpImg(imgContours,biggest,wP,hP)
        #在已获取的a4底图像中继续寻找轮廓
        imgContours2, conts2 = utils.getContours(imgWarp, minArea=2000, filter=4,cThr=[50,50],draw=False)

        if len(conts2) !=0:
            for obj in conts2:
                #绘制四个顶点构成的轮廓
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                #这里没有再进行矫正，所以点的顺序是不正确的，需要手动调用reorder
                nPoints=utils.reorder(obj[2])
                #求两点间距离，求长宽，注意放缩因子
                nW=round(utils.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10,1)
                nH=round(utils.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10,1)
                #绘制箭头
                cv2.arrowedLine(imgContours2,(nPoints[0][0][0],nPoints[0][0][1]),(nPoints[1][0][0],nPoints[1][0][1]),(255,0,255),3,8,0,0.05)
                cv2.arrowedLine(imgContours2,(nPoints[0][0][0],nPoints[0][0][1]),(nPoints[2][0][0],nPoints[2][0][1]),(255,0,255),3,8,0,0.05)
                x,y,w,h=obj[3]
                cv2.putText(imgContours2,'{}cm'.format(nW),(x+30,y-10),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,0,255),thickness=2)
                cv2.putText(imgContours2,'{}cm'.format(nH),(x-70,y+h//2),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,0,255),thickness=2)


        cv2.imshow('A4', imgContours2)

    #进行放缩，可有可无
    img=cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('original',img)
    cv2.waitKey(1)


