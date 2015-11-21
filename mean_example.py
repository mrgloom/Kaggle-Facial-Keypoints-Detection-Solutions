import numpy as np
import pandas as pd
import cv2
import time
import os

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#Compute mean shape using train data
#Compute RMSE and show results

#Results:
#score 4.30414 with drop NaNs
#score 3.96244 with impute NaNs

w= 96 #image width
h= 96 #image height
number_of_keypoints= 15

def ImputeNaNs(df):
	df["left_eye_center_x"].fillna(df["left_eye_center_x"].mean(), inplace=True)
	df["left_eye_center_y"].fillna(df["left_eye_center_y"].mean(), inplace=True)
	df["right_eye_center_x"].fillna(df["right_eye_center_x"].mean(), inplace=True)
	df["right_eye_center_y"].fillna(df["right_eye_center_y"].mean(), inplace=True)
	df["left_eye_inner_corner_x"].fillna(df["left_eye_inner_corner_x"].mean(), inplace=True)
	df["left_eye_inner_corner_y"].fillna(df["left_eye_inner_corner_y"].mean(), inplace=True)
	df["left_eye_outer_corner_x"].fillna(df["left_eye_outer_corner_x"].mean(), inplace=True)
	df["left_eye_outer_corner_y"].fillna(df["left_eye_outer_corner_y"].mean(), inplace=True)
	df["right_eye_inner_corner_x"].fillna(df["right_eye_inner_corner_x"].mean(), inplace=True)
	df["right_eye_inner_corner_y"].fillna(df["right_eye_inner_corner_y"].mean(), inplace=True)
	df["right_eye_outer_corner_x"].fillna(df["right_eye_outer_corner_x"].mean(), inplace=True)
	df["right_eye_outer_corner_y"].fillna(df["right_eye_outer_corner_y"].mean(), inplace=True)
	df["left_eyebrow_inner_end_x"].fillna(df["left_eyebrow_inner_end_x"].mean(), inplace=True)
	df["left_eyebrow_inner_end_y"].fillna(df["left_eyebrow_inner_end_y"].mean(), inplace=True)
	df["left_eyebrow_outer_end_x"].fillna(df["left_eyebrow_outer_end_x"].mean(), inplace=True)
	df["left_eyebrow_outer_end_y"].fillna(df["left_eyebrow_outer_end_y"].mean(), inplace=True)
	df["right_eyebrow_inner_end_x"].fillna(df["right_eyebrow_inner_end_x"].mean(), inplace=True)
	df["right_eyebrow_inner_end_y"].fillna(df["right_eyebrow_inner_end_y"].mean(), inplace=True)
	df["right_eyebrow_outer_end_x"].fillna(df["right_eyebrow_outer_end_x"].mean(), inplace=True)
	df["right_eyebrow_outer_end_y"].fillna(df["right_eyebrow_outer_end_y"].mean(), inplace=True)
	df["nose_tip_x"].fillna(df["nose_tip_x"].mean(), inplace=True)
	df["nose_tip_y"].fillna(df["nose_tip_y"].mean(), inplace=True)
	df["mouth_left_corner_x"].fillna(df["mouth_left_corner_x"].mean(), inplace=True)
	df["mouth_left_corner_y"].fillna(df["mouth_left_corner_y"].mean(), inplace=True)
	df["mouth_right_corner_x"].fillna(df["mouth_right_corner_x"].mean(), inplace=True)
	df["mouth_right_corner_y"].fillna(df["mouth_right_corner_y"].mean(), inplace=True)
	df["mouth_center_top_lip_x"].fillna(df["mouth_center_top_lip_x"].mean(), inplace=True)
	df["mouth_center_top_lip_y"].fillna(df["mouth_center_top_lip_y"].mean(), inplace=True)
	df["mouth_center_bottom_lip_x"].fillna(df["mouth_center_bottom_lip_x"].mean(), inplace=True)
	df["mouth_center_bottom_lip_y"].fillna(df["mouth_center_bottom_lip_y"].mean(), inplace=True)

	return df
	
def LoadTrainData(filename, flDropNaNs= False):
	df = pd.read_csv(filename, header=0) #(7049L, 31L)
	print "Initial train size", df.values.shape

	if(flDropNaNs):
		#drop NaNs
		df = df.dropna() #(2140L, 31L)
		print "After drop NaNs",df.values.shape
	else:
		#impute each column with column mean
		df = ImputeNaNs(df)
		print "After impute NaNs",df.values.shape

	# Extracting images
	df['Image']= df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	X_train= np.vstack(df['Image'].values)
	X_train= X_train.reshape(-1,h,w)
	X_train= X_train.astype(np.uint8)
	print "X_train", X_train.shape

	# Extracting points
	df= df.drop(['Image'], axis=1)
	Y_train= df.values.astype(np.float32)
	print "Y_train",Y_train.shape

	return X_train,Y_train

def LoadTestData(name):
	df = pd.read_csv(name, header=0) #(7049L, 31L)
	print "Initial test size",df.values.shape

	# Extracting Images
	df['Image']= df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	X_test= np.vstack(df['Image'].values)
	X_test= X_test.reshape(-1,h,w)
	X_test= X_test.astype(np.uint8)
	print "X_test", X_test.shape
	
	return X_test

def WritePredictionToCsvInKaggleFormat(Y_pred):		
	nImages= Y_pred.shape[0]
	ImageId=[]
	FeatureName=[]
	for i in range(0,nImages):
		for j in range(0,2*number_of_keypoints):
			ImageId.append(i+1)
			if(j==0):
				FeatureName.append('left_eye_center_x')
			if(j==1):
				FeatureName.append('left_eye_center_y')
			if(j==2):
				FeatureName.append('right_eye_center_x')
			if(j==3):
				FeatureName.append('right_eye_center_y')
			if(j==4):
				FeatureName.append('left_eye_inner_corner_x')
			if(j==5):
				FeatureName.append('left_eye_inner_corner_y')
			if(j==6):
				FeatureName.append('left_eye_outer_corner_x')
			if(j==7):
				FeatureName.append('left_eye_outer_corner_y')
			if(j==8):
				FeatureName.append('right_eye_inner_corner_x')
			if(j==9):
				FeatureName.append('right_eye_inner_corner_y')
			if(j==10):
				FeatureName.append('right_eye_outer_corner_x')
			if(j==11):
				FeatureName.append('right_eye_outer_corner_y')
			if(j==12):
				FeatureName.append('left_eyebrow_inner_end_x')
			if(j==13):
				FeatureName.append('left_eyebrow_inner_end_y')
			if(j==14):
				FeatureName.append('left_eyebrow_outer_end_x')
			if(j==15):
				FeatureName.append('left_eyebrow_outer_end_y')
			if(j==16):
				FeatureName.append('right_eyebrow_inner_end_x')
			if(j==17):
				FeatureName.append('right_eyebrow_inner_end_y')
			if(j==18):
				FeatureName.append('right_eyebrow_outer_end_x')
			if(j==19):
				FeatureName.append('right_eyebrow_outer_end_y')
			if(j==20):
				FeatureName.append('nose_tip_x')
			if(j==21):
				FeatureName.append('nose_tip_y')
			if(j==22):
				FeatureName.append('mouth_left_corner_x')
			if(j==23):
				FeatureName.append('mouth_left_corner_y')
			if(j==24):
				FeatureName.append('mouth_right_corner_x')
			if(j==25):
				FeatureName.append('mouth_right_corner_y')
			if(j==26):
				FeatureName.append('mouth_center_top_lip_x')
			if(j==27):
				FeatureName.append('mouth_center_top_lip_y')
			if(j==28):
				FeatureName.append('mouth_center_bottom_lip_x')
			if(j==29):
				FeatureName.append('mouth_center_bottom_lip_y')

	df_a= pd.DataFrame()
	df_a['ImageId']= ImageId
	df_a['FeatureName']= FeatureName
	df_a["Location"]= Y_pred.reshape(-1,1)

	df_b= pd.read_csv('kaggle_data/IdLookupTable.csv',header=0)

	df_b = df_b.drop('Location',axis=1)
	merged = df_b.merge(df_a, on=['ImageId','FeatureName'] )
	
	if not os.path.exists('output'):
		os.makedirs('output')
	
	merged.to_csv('output/kaggle_submission.csv', index=0, columns = ['RowId','Location'] )
	
def Predict(X_test, Y_train, flShowKeypoints= False):
	#Compute mean model
	mean_model= np.mean(Y_train,axis=0)
	mean_model= mean_model.reshape(number_of_keypoints,2)
	print "mean_model",mean_model.shape
		
	Y_pred= np.zeros((X_test.shape[0],2*number_of_keypoints),np.float32)
	for i in range(0,X_test.shape[0]):
		if(flShowKeypoints):
			print i
			img= X_test[i,:,:]
			#show prediction
			for point in mean_model:
				img[round(point[1])][round(point[0])] = 255

			cv2.imshow("mean model", img)
			cv2.waitKey(0)
		
		Y_pred[i,:]= mean_model.reshape(1,2*number_of_keypoints)
	
	WritePredictionToCsvInKaggleFormat(Y_pred)
	
def ComputeRMSE(X,Y):
	#train/test split 80%/20%
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	print "X_train",X_train.shape
	print "Y_train",Y_train.shape
	print "X_test",X_test.shape
	print "Y_test",Y_test.shape
	
	#Compute mean model
	mean_model= np.mean(Y_train,axis=0)
	mean_model= mean_model.reshape(number_of_keypoints,2)
	print "mean_model",mean_model.shape

	Y_pred= np.zeros((X_test.shape[0],2*number_of_keypoints),np.float32)
	for i in range(0,X_test.shape[0]):		
		Y_pred[i,:]= mean_model.reshape(1,2*number_of_keypoints)
	
	RMSE= sqrt(mean_squared_error(Y_test, Y_pred))
	
	print RMSE
	
#--------------------------------------------------------------------------
t = time.time()
X_train,Y_train= LoadTrainData('kaggle_data/training.csv', flDropNaNs= False)
X_test= LoadTestData('kaggle_data/test.csv')
Predict(X_test,Y_train, flShowKeypoints= False)
print "Done in %f s" % (time.time()-t)

#X_train,Y_train= LoadTrainData('kaggle_data/training.csv', flDropNaNs= True)
#ComputeRMSE(X_train,Y_train)