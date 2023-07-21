#最大轮廓检测

import cv2
import numpy as np
import math

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
    #转灰度图
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #模糊处理，这里使用高斯滤波
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)

    #边缘检测，这是使用Canny算法
    imgCanny=cv2.Canny(imgBlur,cThr[0],cThr[1])

    #膨胀+腐蚀，加强边缘信息
    kernel=np.ones((5,5))
    imgDial=cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre=cv2.erode(imgDial,kernel,iterations=2)

    if showCanny:
        cv2.imshow('Canny',imgThre)
        cv2.waitKey(0)

    #从图像中寻找轮廓，这里我们要找所有的最外层的轮廓，每个轮廓是一个点的序列
    contours,hiearchy=cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []  # 最终的轮廓序列
    #根据围出筛选轮廓
    for i in contours:
        #计算围出面积
        area=cv2.contourArea(i)
        #判断面积

        if area>minArea:
            #轮廓求周长
            peri=cv2.arcLength(i,True)
            #将曲线轮廓近似为直线，中间的参数为近似精度阈值，以周长为基准，返回值为近似之后的顶点序列，若矩形则为4个顶底，用于后面透视变换
            approx=cv2.approxPolyDP(i,0.02*peri,True)
            #根据轮廓返回左上角坐标以及长宽
            bbox=cv2.boundingRect(approx)
            #过滤一些指定顶点数量的轮廓，filter为指定的顶点数
            if filter>0:
                if len(approx)==filter:
                    #轮廓包含顶点数量，所围面积，顶点序列，轮廓参数，近似前的点序列
                    finalContours.append([len(approx),area,approx,bbox,i])
            #不需要过滤
            else:
                finalContours.append([len(approx), area, approx, bbox, i])

    #将列表依照area进行递减排序，中间使用了lambda表达式来取area
    finalContours=sorted(finalContours,key=lambda x:x[1],reverse=True)

    #绘制轮廓
    if draw:
        for con in finalContours:
            cv2.drawContours(img,con[4],-1,(0,0,255),5)

    return img,finalContours

#通过轮廓检测的到的顶点并不一定按照顺序排列，这里我们需要把顺序统一为依次左上右上左下右下
def reorder(myPoints):
    print(myPoints.shape)
    #新建一个点序列矩阵用于保存排序后的正确的点序列
    myPointsNew=np.zeros_like(myPoints)

    #这里输出发现为三维数组，多出来一个维度，且维数为1，通过reshape进行清除
    myPoints=myPoints.reshape((4,2))

    #求出每个点横纵坐标和，用于判断左上角和右下角
    add=myPoints.sum(1)

    #通过add的最值索引，找到对应点的下标，放在new的矩阵中
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    #对每个点的横纵坐标作差，用于判断右上角和左下角
    diff=np.diff(myPoints,axis=1)

    #通过add的最值索引，找到对应点的下标，放在new的矩阵中
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

#根据轮廓定点进行图像矫正
def warpImg(img,points,w,h,pad=20):
    # print(points)
    points=reorder(points)

    #使用透视变换矫正图像
    pts1=np.float32(points)#原图四点
    pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])#新图四点
    matrix=cv2.getPerspectiveTransform(pts1,pts2)#求变换矩阵
    imgWarp=cv2.warpPerspective(img,matrix,(w,h))#开始变换

    #去除边界的误差，上下左右都减去pad长度的像素
    imgWarp=imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]

    return imgWarp

#通过两点的坐标计算两点距离
def findDis(pts1,pts2):
    return math.sqrt((abs(pts1[0]-pts2[0])**2+abs(pts1[1]-pts2[1])**2))