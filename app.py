#import sys
from cmath import rect
from pickle import FALSE
from sre_parse import WHITESPACE
from tkinter.tix import IMAGE
from matplotlib import testing
from matplotlib.cbook import is_writable_file_like
import pygame, sys
from pygame.locals import *
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
from sympy import Predicate

WIN_X=1140
WIN_Y=880
WHITE=(255,255,0)
BLACK=(0,0,0)
GREEN=(255,0,0)

IMAGESAVE=False
imag_cnt=1
PREDICT=True

MODEL=load_model("bestmodel.h5")

LABELS={0:"ZERO",1:"ONE",
2:"TWO",3:"THREE",
4:"FOUR",5:"FIVE",
6:"SIX",7:"SEVEN",
8:"EIGHT",9:"NINE"}

pygame.init()
FONT=pygame.font.Font("Cascadia.ttf",20)



DISPLAYSURF=pygame.display.set_mode((WIN_X,WIN_Y))

WHILE_INT = DISPLAYSURF.map_rgb(WHITE)

pygame.display.set_caption("DURGANSH's Playground")
iswriting=False


number_xcord=[]
number_ycord=[]
BOUNDRYINC=5

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
            
        if event.type==MOUSEMOTION and iswriting :
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)


            
        if event.type==MOUSEBUTTONDOWN:
            iswriting=True

        if event.type==MOUSEBUTTONUP:
            iswriting=False
            number_xcord=sorted(number_xcord)
            number_ycord=sorted(number_ycord)

            rect_min_x,rect_max_x=max(number_xcord[0]-BOUNDRYINC,0),min(WIN_X,number_xcord[-1]+BOUNDRYINC)
            rect_min_y,rect_max_y=max(0,number_ycord[0]-BOUNDRYINC),min(number_ycord[-1]+BOUNDRYINC,WIN_X)

            number_xcord=[]
            number_ycord=[]

            img_arr=np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
            #img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("images/image-{%d}.png" % image_cnt, img_arr)
                imag_cnt +=1
            
            if PREDICT:

                image=cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10),'constant',constant_values=0)
                image=cv2.resize(image,(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                textSurface=FONT.render(label,True,GREEN,WHITE)
                textRecObj=textSurface.get_rect()
                textRecObj.left,textRecObj.bottom=rect_min_x,rect_min_y

                DISPLAYSURF.blit(textSurface,textRecObj)

            if event.type==KEYDOWN:
                if event.unicode=="n":
                    DISPLAYSURF.fill(BLACK)


        pygame.display.update()






