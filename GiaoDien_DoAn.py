from tkinter import *
from tkinter import filedialog, Tk, Label
import os
import tkinter as tk
from PIL import Image, ImageTk

# Import thư viện
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import cv2
import math

# Import thư viện
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

class Root(Tk):
	def __init__(self):
		super(Root, self).__init__()
		self.title("Do An")
		self.minsize(1000, 900)
		self.configure(background='#4D4D4D')
		self.buttonOpenImage()
		self.buttonCalculateScore()
		#self.LabelScore()
		file_path_root = ''

	def OpenImage(self):
		self.file_path_root = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(
		("JPG file", "*.jpg"), ("PNG file", "*.png"), ("All files", "*.*")))
		image = Image.open(self.file_path_root)
		image = image.resize((600,700), Image.ANTIALIAS)
		image = image.rotate(-90)
		img = ImageTk.PhotoImage(image)
		labelImage = Label(self, image=img)
		labelImage.image = img
		labelImage.place(x=80,y=85)

	def CalculateScore(self): #code thuat toan o day
		result = ChamDiem(self.file_path_root)
		dap_an = ['C']*40
		diem = 0
		for i, correct_ans in enumerate(dap_an):
			if correct_ans == result[i]:
				diem += 10/len(dap_an)
		labelScore = Label(self, text="ĐIỂM SỐ: "+ str(diem), bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelScore.place(x=770, y=700)
		lblKQ = Label(self,text="KẾT QUẢ", bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		lblKQ.place(x=710,y=85)
		lbl_result1 = "\n"
		lbl_result2 = "\n"
		for i in range(20):
			lbl_result1 = lbl_result1 + str(i+1) + '.' + result[i] + "\n"
		for i in range(20,40):
			lbl_result2 = lbl_result2 + str(i+1) + '.' + result[i] + "\n"
		labelKetQua1 = Label(self, text=lbl_result1, bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelKetQua1.place(x=700, y=135)
		labelKetQua2 = Label(self, text=lbl_result2, bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelKetQua2.place(x=770, y=135)
		lblDA = Label(self,text="ĐÁP ÁN", bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		lblDA.place(x=870,y=85)
		lbl_dapan1 = "\n"
		lbl_dapan2 = "\n"
		for i in range(20):
			lbl_dapan1 = lbl_dapan1 + str(i+1) + '.' + dap_an[i] + "\n"
		for i in range(20,40):
			lbl_dapan2 = lbl_dapan2 + str(i+1) + '.' + dap_an[i] + "\n"
		labelDapAn1 = Label(self, text=lbl_dapan1, bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelDapAn1.place(x=850, y=135)
		labelDapAn2 = Label(self, text=lbl_dapan2, bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelDapAn2.place(x=920, y=135)

	def buttonOpenImage(self):
		btn_LoadImage = Button(self, text="Chọn Ảnh", width=25, height=3, command=self.OpenImage)
		btn_LoadImage.grid(row=3, column=2, padx=130, pady=25)

	def buttonCalculateScore(self):
		btn_CalculateScore = Button(self, text="Chấm Điểm", width=25, height=3, command=self.CalculateScore)
		btn_CalculateScore.grid(row=3, column=3, padx=0, pady=25)

	def LabelScore(self):
		labelScore = Label(self, text="Score: ", bg='#fff', fg='#f00', pady=10, padx=10, font=13)
		labelScore.place(x=1100, y=30)

def ChamDiem(file_path):
    

	path = file_path

	

	# Khởi tạo giá trị huấn luyện
	BATCH_SIZE = 32
	IMG_SIZE = 28
	N_CLASSES = 4
	LR = 0.001
	N_EPOCHS = 20

	# Xây dựng mô hình
	network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

	network = conv_2d(network, 32, 3, activation='relu') #2
	network = max_pool_2d(network, 2) #3

	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)

	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)

	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)

	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)

	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)

	network = fully_connected(network, 1024, activation='relu') #4
	network = dropout(network, 0.8) #5

	network = fully_connected(network, N_CLASSES, activation='softmax')#6
	network = regression(network)

	model = tflearn.DNN(network) #7

	# Load mô hình đã lưu
	model.load('E:/NhanDang/PhieuDienTracNghiem/model/letter.tflearn')

	# 2.Detect ảnh
	# nhập ảnh
	img = cv2.imread(path, 3)

	form = img.copy()
	gray = cv2.cvtColor(form, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	#cv2.imshow('thresh',thresh)
	h = form.shape[0]
	w = form.shape[1]
	#print(form.shape)

	horizal = thresh
	vertical = thresh

	scale_height = 20 #Scale này để càng cao thì số dòng dọc xác định sẽ càng nhiều
	scale_long = 15

	long = int(img.shape[1]/scale_long)
	height = int(img.shape[0]/scale_height)

	horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
	horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
	horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
	vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
	vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

	mask = vertical + horizal
	#cv2.imshow('mask',mask)

	contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	max = -1
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if cv2.contourArea(cnt) > max:
			x_max, y_max, w_max, h_max = x, y, w, h
			max = cv2.contourArea(cnt)

	table =  img[y_max:y_max+h_max, x_max:x_max+w_max]
	table1 = mask[y_max:y_max+h_max, x_max:x_max+w_max]

	h = table.shape[0]
	w = table.shape[1]
	#print(h ,w)

	#cv2.imshow('table1',table1)

	#print(np.pi/180)
	x1 = np.pi/180*88
	x2 = np.pi/180*92
	x3 = np.pi/180*(-5)
	x4 = np.pi/180*(5)

	#ham sap xep cac duong theo thu tu
	def SortLine(l):
		for i in range(len(l)):
			for j in range(len(l)-1):
				if(l[j][0][0] > l[j+1][0][0]):
					temp = l[j][0][0]
					l[j][0][0] = l[j+1][0][0]
					l[j+1][0][0] = temp
		return l

	lines = cv2.HoughLines(table1, 1, np.pi / 180, 700, None, 0, 0) #line : vector luu ket qua duoi dang (p, O)-  p la khoang cach, 0 la goc tao voi duong thang
	#print(lines)
	#print(len(lines))

	lines2 = []
	for line in lines:
		# if((line[0][1]>x1 and line[0][1]<x2) or (line[0][1]>x3 and line[0][1]<x4)):
		if((line[0][1]>x1 and line[0][1]<x2)):
			lines2.append(line)
	#print(len(lines2))
	#print(lines2)

	h = table.shape[0] #chieu cao
	w = table.shape[1] #chieu rong
	#print(h ,w)

	lines3 = []   # line ngang
	i = 0
	while(i < h):
		j = 0
		while(j < len(lines2)):
			if(lines2[j][0][0] == int(i*math.sin(lines2[j][0][1]))+1):
				if(i > 0.03*h and i < 0.97*h):
					lines3.append(lines2[j])
				i = i + int(h/len(lines2))
				continue
			j = j+1
		i = i + 1
	lines3.append([[h,np.pi/2]])
	lines3 = SortLine(lines3)
	#print(len(lines3))

	lines = cv2.HoughLines(table1, 1, np.pi / 180, 500, None, 0, 0) #line : vector luu ket qua duoi dang (p, O)-  p la khoang cach, 0 la goc tao voi duong thang

	lines4 = []
	for line in lines:
		# if((line[0][1]>x1 and line[0][1]<x2) or (line[0][1]>x3 and line[0][1]<x4)):
		if((line[0][1]>x3 and line[0][1]<x4)):
			lines4.append(line)
	#print(len(lines4))
	#rint(lines4)

	lines5 = []   # đường dọc
	i = 0
	while(i < w):
		j = 0
		while(j < len(lines4)):
			if(lines4[j][0][0] == int(i*math.cos(lines4[j][0][1]))+1):
				if(i > 0.02*w and i < 0.98*w):
					lines5.append(lines4[j])
				i = i + int(h/(2*len(lines4)))
				continue
			j = j+1
		i = i + 1
	lines5.append([[w,0]])
	lines5 = SortLine(lines5)
	#print(len(lines5))

	lines6 = lines3 + lines5

	cdst = table.copy()
	cdstP = table.copy()

	if lines6 is not None:
		for i in range(0, len(lines6)):
			rho = lines6[i][0][0]  # p
			theta = lines6[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
			pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))
			cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
	
	line1 = lines3[0][0]
	line2 = lines3[1][0]

	line3 = lines5[0][0]
	line4 = lines5[1][0]

	#print(line1, line2, line3, line4)
	#print(line1[0])
	x1 = int((line3[0] - line1[0]*math.cos(line3[1])/math.cos(line1[1])) / (math.sin(line3[1]) - math.sin(line1[1])*math.cos(line3[1])/math.cos(line1[1])))
	x2 = int((line4[0] - line1[0]*math.cos(line4[1])/math.cos(line1[1])) / (math.sin(line4[1]) - math.sin(line1[1])*math.cos(line4[1])/math.cos(line1[1])))

	x3 = int((line3[0] - line2[0]*math.cos(line3[1])/math.cos(line2[1]))/(math.sin(line3[1]) - math.sin(line2[1])*math.cos(line3[1])/math.cos(line2[1])))
	x4 = int((line4[0] - line2[0]*math.cos(line4[1])/math.cos(line2[1]))/(math.sin(line4[1]) - math.sin(line2[1])*math.cos(line4[1])/math.cos(line2[1])))

	y1 = int(line1[0]+line3[0] - x1*(math.cos(line1[1]) + math.cos(line3[1]))/(math.sin(line1[1])+math.sin(line3[1])))
	y2 = int(line1[0]+line4[0] - x2*(math.cos(line1[1]) + math.cos(line4[1]))/(math.sin(line1[1])+math.sin(line4[1])))
	y3 = int(line2[0]+line3[0] - x3*(math.cos(line2[1]) + math.cos(line3[1]))/(math.sin(line2[1])+math.sin(line3[1])))
	y4 = int(line2[0]+line4[0] - x4*(math.cos(line2[1]) + math.cos(line4[1]))/(math.sin(line2[1])+math.sin(line4[1])))

	#print([x1,y1], [x2,y2], [x3,y3], [x4,y4])
	#table1 = mask[y_max:y_max+h_max, x_max:x_max+w_max]
	test  = table[x1:x4,y1:y4]
	#cv2.imshow('test',test)

	def Crop(table,line1,line2,line3,line4):  # line1, line2 duong ngang, line3, line4 duong doc
		x1 = int((line3[0] - line1[0]*math.cos(line3[1])/math.cos(line1[1])) / (math.sin(line3[1]) - math.sin(line1[1])*math.cos(line3[1])/math.cos(line1[1])))
		x2 = int((line4[0] - line2[0]*math.cos(line4[1])/math.cos(line2[1]))/(math.sin(line4[1]) - math.sin(line2[1])*math.cos(line4[1])/math.cos(line2[1])))

		y1 = int(line1[0]+line3[0] - x1*(math.cos(line1[1]) + math.cos(line3[1]))/(math.sin(line1[1])+math.sin(line3[1])))
		y2 = int(line2[0]+line4[0] - x2*(math.cos(line2[1]) + math.cos(line4[1]))/(math.sin(line2[1])+math.sin(line4[1])))
		img_crop = table[x1 + 10:x2-10,y1 + int(0.03*(y2-y1)):y2 - int(0.03*(y2-y1))]
		return img_crop

	line1 = lines3[19][0]
	line2 = [h,np.pi/2]
	#line2 = lines3[2][0]

	line3 = lines5[2][0]
	#line4 = lines5[1][0]
	line4 = [w, 0]
	crop = Crop(table,line1,line2,line3,line4)
	#print(crop)
	#cv2_imshow(crop)

	#Khung đáp án
	imgAnswer = []
	listAnswer = []
	for i in range(len(lines3)-1):
		imgAnswer = Crop(table,lines3[i][0],lines3[i+1][0],lines5[0][0],lines5[1][0])
		listAnswer.append(imgAnswer)
	for i in range(len(lines3)-1):
		imgAnswer = Crop(table,lines3[i][0],lines3[i+1][0],lines5[2][0],lines5[3][0])
		listAnswer.append(imgAnswer)

	def convert_to_binary(img_grayscale, thresh=100):
		thresh, img_binary = cv2.threshold(img_grayscale, thresh, maxval=255, type=cv2.THRESH_BINARY)
		return img_binary

	img1 = listAnswer[1].copy()
	gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	# blur = convert_to_binary(gray,150)
	blur = cv2.Canny(gray,100,150)
	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	#ret, thresh = cv2.threshold(gray, 150 ,255,0)
	#cv2.imshow('thresh',thresh)

	contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	answer = []
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if  (w > 30 and h > 30 and cv2.contourArea(cnt)>100):
			answer.append(cnt)

	x, y, w, h = cv2.boundingRect(answer[0])
	img_temp = img1[y:y+h,x:x+w+10]
	#cv2.imshow('img_temp',img_temp)

	cnt = answer[0]
	#print(cv2.boundingRect(cnt))
	x, y, w, h = cv2.boundingRect(cnt)
	img2 = img1[y:y+h,x:x+w]
	#cv2.imshow('img2',img2)

	#hàm lấy đáp án
	anh = 0
	answer = []
	for img in listAnswer:
		#print("anh thu : ", anh)
		anh = anh +1
		temp = img.copy()
		#cv2.imshow('temp',temp)
		gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		#blur = convert_to_binary(gray,110)
		blur = cv2.Canny(gray,100,150)
		thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		ans = []
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			if  (w > 30 and h > 30 and cv2.contourArea(cnt) > 100):
				ans.append(cnt)
		if(len(ans)>0):
			ans_img = []
			for cnt in ans:
				x, y, w, h = cv2.boundingRect(cnt)
				img_ans = thresh[y:y+h,x-20:x+w+20]
				#img_ans = convert_to_binary(img_ans,100)
				#img_ans = cv2.adaptiveThreshold(img_ans,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
				img_ans = cv2.resize(img_ans, (28,28))
				#print(cv2.boundingRect(cnt))
			
			#cv2.imshow('img_ans',img_ans)
			ans_img.append(img_ans)
		else:
			#print("khong co contour")
			ans_img = []
		answer.append(ans_img)

	# Dự đoán kí tự câu trả lời của thí sinh
	res_predict = []

	for ans in answer:
		if len(ans)!=0:
			ans = ans[0].reshape(-1,28,28,1)
			res_predict.append(np.argmax(model.predict(ans)))
		else:
			res_predict.append(-1)

	print(res_predict)

	# Lưu đáp án của thí sinh
	letter = ['A', 'B', 'C', 'D']
	result = []
	for r in res_predict:
		if r != -1:
			result.append(letter[r])
		else:
			result.append('X')

	print(result)
	print(len(result))

	return result

if __name__ == '__main__':
	root = Root()
	root.mainloop()