from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
import cv2
from numpy.fft import fft2, fftshift
import numpy as np
from prometheus_client import Counter
from viewer import Viewer
import numpy as np


class segmentation(qtw.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("src/ui/segmentation.ui", self)
        self.original_image = Viewer()
        self.image_layout.addWidget(self.original_image)
        self.Binary_image = Viewer()
        self.Binary_image_layout.addWidget(self.Binary_image)
        self.segmentation = Viewer()
        self.segmentation_layout.addWidget(self.segmentation)
        self.kernal_size=3
        self.local_means=[]
        self.local_varience=[]
        self.local_mean_var=[]
        self.counter=0
        self.dict={}
        self.valid=[]
        self.segmentation_btn.clicked.connect(self.segmentate)
        self.threshold_value.valueChanged.connect(self.binarizing_image)

    def load_original_image(self, image_path):
        self.Gray_img = cv2.imread(image_path,0)
        self.imag1=self.Gray_img
        self.draw(self.original_image,self.Gray_img)
        self.binarizing_image()
        self.clear(self.segmentation)
        self.size=self.Gray_img.shape
        self.num_kernals_rows=int((self.size[0]-self.size[0]%self.kernal_size)/self.kernal_size)
        self.num_kernals_colums=int((self.size[1]-self.size[1]%self.kernal_size)/self.kernal_size)
        self.center=int((self.num_kernals_colums*self.num_kernals_rows)/2)
        

    def draw(self,layout,image):
        layout.draw_image(image)

    def clear(self,layout):
        layout.clear_canvans()

    def binarizing_image(self):
        _,self.Gray_img_binary=cv2.threshold(self.imag1,self.threshold_value.value(),255,cv2.THRESH_BINARY)
        self.thershold_value.setText(str(self.threshold_value.value()))  
        self.draw(self.Binary_image ,self.Gray_img_binary)
        self.Gray_img=self.Gray_img_binary

    def segmentate(self):
        self.thrshold_1=5/10
        self.radius = 150/100
        self.local_mean_std()
        self.iteration(self.local_means[self.center],self.local_varience[self.center])
        print(self.local_means[self.center],self.local_varience[self.center])
        self.reverse()
        
    def local_mean_std(self):
        self.dict={} 
        self.local_means=[]
        self.local_varience=[]
        self.counter=0
        for j in range(self.num_kernals_colums):
            for i in range(self.num_kernals_rows):
               self.calculate([i*self.kernal_size,(1+i)*self.kernal_size],[j*self.kernal_size,(1+j)*self.kernal_size]) 
        self.calculate([self.num_kernals_rows*self.kernal_size,self.size[0]-1],[self.num_kernals_colums*self.kernal_size,self.size[1]])  
    def calculate(self,start_end_row,start_end_colum):
        sum_I=0
        sum_Ipow2=0
        self.dict[self.counter]=[start_end_row,start_end_colum]
        self.counter=self.counter+1
        for x in range(start_end_row[0],start_end_row[1]): #rows 
            for y in range(start_end_colum[0],start_end_colum[1]): #columns
                sum_I=sum_I+self.Gray_img[x,y]
                sum_Ipow2=sum_Ipow2+self.Gray_img[x,y]**2
        mean=sum_I/self.kernal_size**2
        var=np.sqrt((sum_Ipow2/self.kernal_size**2)-(mean**2))
        self.local_varience.append(var)
        self.local_means.append(mean)
        # self.local_mean_var.append({self.counter:[mean,var]})

    def iteration(self,mean0,var0):
        self.valid=[]
        for i in range(len(self.local_means)):
            ep=(self.local_means[i]-mean0)**2+(self.local_varience[i]-var0)**2
            if(ep<=self.radius**2):
                self.valid.append(i)
        sum1=0
        sum2=0
        for i in self.valid:
            sum1=sum1+self.local_means[i]
            sum2=sum2+self.local_varience[i]
        mean =sum1/len(self.valid) 
        var=sum2/len(self.valid) 
        if(((mean-mean0)**2+(var-var0))<=self.thrshold_1**2):
            return
        else:
            if (mean0-mean)>0:
                for i in self.valid:
                   self.local_means[i]=self.local_means[i]-(mean0-mean) 
            else:
                for i in self.valid:
                   self.local_means[i]=self.local_means[i]+(mean-mean0)   
            if (var0-var)>0:
                for i in self.valid:
                   self.local_varience[i]=self.local_varience[i]-(var0-var) 
            else:
                for i in self.valid:
                   self.local_varience[i]=self.local_varience[i]+(var-var0)                
            self.iteration(mean,var)

    
    def reverse(self):
        for i in self.valid:
            rows=self.dict[i][0]
            cols=self.dict[i][1]
            for x in range(rows[0],rows[1]): #rows 
                for y in range(cols[0],cols[1]): #columns
                    self.Gray_img[x,y]=90
        self.clear(self.segmentation)
        self.draw(self.segmentation,self.Gray_img)            



            


          

                  