#httpscikit-learn.orgstableauto_examplestreeplot_tree_regression_multioutput.html
#httpscikit-learn.orgstableauto_examplesplot_multioutput_face_completion.html

#add preprocessing, normalization/whitening
#add features
#http://scikit-image.org/docs/dev/auto_examples/plot_local_binary_pattern.html

#mirror don't work wright!

#Solution using pure pixels and sklearn DecisionTreeRegressor

#score 4.38436 with drop NaNs
#score 3.91109 with inpute Nans

#?
#score 3.97049 if use opencv's histogram equalization
#score 3.93624 if use numpy histogram equalization

import numpy as np
import pandas as pd
import cv2
import time
import os
from math import sqrt

from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


w= 96 #image width
h= 96 #image height
number_of_keypoints= 15

#def HistogramEqualization(X_row):
#	img= X_row.reshape(h,w).astype(np.uint8)
#	img= cv2.equalizeHist(img)
#	
#	return img.reshape(1,h*w).astype(np.float32)

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def HistogramEqualization(X_row):

	X_row= image_histogram_equalization(X_row)[0]
	
	return X_row
	
def NormalizeData(X):
	X -= np.mean(X, axis = 0)
	X /= np.std(X, axis = 0)
	
	return X
	
def GetDecomposer(X, nComponents=100):
	pca = RandomizedPCA(n_components=nComponents)
	pca.fit(X)

	return pca

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

def MirrorImages(X):
	Xaug= np.copy(X)
	for i in range(0,X.shape[0]):
		img= X[i,:].reshape(h,w)
		img= cv2.flip(img,1)
		Xaug[i,:]= img.reshape(1, h*w)

	return Xaug

def MirrorLandmarks(Y):
	Yaug= np.copy(Y)
	for i in range(0,Y.shape[0]):
		for j in range(0,2*number_of_keypoints):
			if(j%2==0):
				Yaug[i,j]= w-Y[i,j]
			else:
				Yaug[i,j]= Y[i,j]
	return Yaug

def VisuallyInspect(X,Y):
	X= X.reshape(-1,h,w)
	for i in range(0,X.shape[0]):
		img= X[i,:,:].astype(np.uint8)
		img= cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		for k in range(0,number_of_keypoints):
			cv2.circle(img,(int(Y[i,k]),int(Y[i,k+1])),1,(0,0,255))
		cv2.imshow("image",img)
		cv2.waitKey(0)

def LoadTrainData(filename, flDropNaNs= False, flUseHistogramEqualization= True, flUseDataNormalization= True, flUseDataAugmentation= False, flVisuallyInspect= False):
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

	X_train= X_train.astype(np.float32)
	
	if(flUseHistogramEqualization):
		for i in range(X_train.shape[0]):
			X_train[i,:]= HistogramEqualization(X_train[i,:])
			
	#if(flUseDataNormalization):
	#	X_train= NormalizeData(X_train)
	
	print "X_train", X_train.shape

	if(flUseDataAugmentation):
		X_train_aug= MirrorImages(X_train)
		X_train= np.vstack((X_train,X_train_aug))
		print "X_train", X_train.shape
	
	# Extracting points
	df= df.drop(['Image'], axis=1)
	Y_train= df.values.astype(np.float32)
	print "Y_train",Y_train.shape
	
	if(flUseDataAugmentation):
		Y_train_aug= MirrorLandmarks(Y_train)
		Y_train= np.vstack((Y_train,Y_train_aug))
		print "Y_train", Y_train.shape
	
	if(flVisuallyInspect):
		VisuallyInspect(X_train,Y_train)#something wrong here
		
	return X_train,Y_train
	
def LoadTestData(filename, flUseHistogramEqualization=True, flUseDataNormalization=True):
	df = pd.read_csv(filename, header=0) #(1783, 2)
	print "Initial test size",df.values.shape

	# Extracting Images
	df['Image']= df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	X_test= np.vstack(df['Image'].values)
	
	X_test= X_test.astype(np.float32)
		
	if(flUseHistogramEqualization):
		for i in range(X_test.shape[0]):
			X_test[i,:]= HistogramEqualization(X_test[i,:])
		
	#maybe we need use mean and std from train?
	#if(flUseDataNormalization):
	#	X_test= NormalizeData(X_test)
		
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
	
def Predict(X_train, Y_train, X_test, flUsePCA=False):
	# Fit regression model
	regr= DecisionTreeRegressor(max_depth=8) #TODO: need to crossvalidate parameteres

	if(flUsePCA):
		pca= GetDecomposer(X_train, nComponents=23) #TODO: need to crossvalidate parameteres
		X_train= pca.fit_transform(X_train)
		print "X_train: ", X_train.shape

	regr.fit(X_train, Y_train)

	if(flUsePCA):
		X_test= pca.fit_transform(X_test)
		print "X_test: ", X_test.shape

	Y_pred= regr.predict(X_test)
	
	return Y_pred
	
def ComputeRMSE(X,Y):
	#train/test split 80%/20%
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
	print "X_train",X_train.shape
	print "Y_train",Y_train.shape
	print "X_test",X_test.shape
	print "Y_test",Y_test.shape
	
	Y_pred= Predict(X_train,Y_train,X_test)
	
	RMSE= sqrt(mean_squared_error(Y_test, Y_pred))
	
	print RMSE
	
#-----------------------------------------------------------------------------------------------------------------------
#1
t = time.time()

X_train,Y_train= LoadTrainData('kaggle_data/training.csv', flDropNaNs= False, flUseDataAugmentation= False, flVisuallyInspect= False)
X_test= LoadTestData('kaggle_data/test.csv')
Y_pred= Predict(X_train,Y_train,X_test)
WritePredictionToCsvInKaggleFormat(Y_pred)

print "Done in %f s" % (time.time()-t)


#2
#t = time.time()

#X_train,Y_train= LoadTrainData('kaggle_data/training.csv', flDropNaNs= True)
#ComputeRMSE(X_train,Y_train)

#print "Done in %f s" % (time.time()-t)