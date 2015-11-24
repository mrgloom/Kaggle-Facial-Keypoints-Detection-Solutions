import numpy as np
import pandas as pd
import cv2
import stasm

#Read images from .csv and apply stasm to detect landmarks
#Compute RMSE and show results

w= 96 #image width
h= 96 #image height
number_of_keypoints= 15

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
	X= np.vstack(df['Image'].values)
	print "X", X.shape
	X= X.reshape(-1,h,w)
	X= X.astype(np.uint8)
	print "X", X.shape

	# Extracting points
	df= df.drop(['Image'], axis=1)
	Y= df.values.astype(np.float32)
	print "Y",Y.shape

	return X,Y

def LoadTestData(name):
	df = pd.read_csv(name, header=0) #(1783, 2)
	print "Initial test size",df.values.shape

	# Extracting Images
	df['Image']= df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
	
	X= np.vstack(df['Image'].values)
	print "X", X.shape
	X= X.reshape(-1,h,w)
	X= X.astype(np.uint8)
	print "X", X.shape
	
	return X
	
def stasm_77_to_kaggle_15(stasm_keypoints_77):
	#http://www.milbo.org/stasm-files/stasm4.pdf
	kaggle_keypoints_15= np.zeros((number_of_keypoints,2),np.float32)
	
	#left_eye_center
	kaggle_keypoints_15[0,:]= stasm_keypoints_77[38,:]
	#right_eye_center
	kaggle_keypoints_15[1,:]= stasm_keypoints_77[39,:]
	#left_eye_inner_corner
	kaggle_keypoints_15[2,:]= stasm_keypoints_77[30,:]
	#left_eye_outer_corner
	kaggle_keypoints_15[3,:]= stasm_keypoints_77[34,:]
	#right_eye_inner_corner
	kaggle_keypoints_15[4,:]= stasm_keypoints_77[40,:]
	#right_eye_outer_corner
	kaggle_keypoints_15[5,:]= stasm_keypoints_77[44,:]
	#left_eyebrow_inner_end
	kaggle_keypoints_15[6,:]= stasm_keypoints_77[21,:]
	#left_eyebrow_outer_end
	kaggle_keypoints_15[7,:]= stasm_keypoints_77[18,:]
	#right_eyebrow_inner_end
	kaggle_keypoints_15[8,:]= stasm_keypoints_77[22,:]
	#right_eyebrow_outer_end
	kaggle_keypoints_15[9,:]= stasm_keypoints_77[25,:]
	#nose_tip
	kaggle_keypoints_15[10,:]= stasm_keypoints_77[52,:]
	#mouth_left_corner
	kaggle_keypoints_15[11,:]= stasm_keypoints_77[59,:]
	#mouth_right_corner
	kaggle_keypoints_15[12,:]= stasm_keypoints_77[65,:]
	#mouth_center_top_lip
	kaggle_keypoints_15[13,:]= stasm_keypoints_77[62,:]
	#mouth_center_bottom_lip	
	kaggle_keypoints_15[14,:]= stasm_keypoints_77[74,:]
	
	return kaggle_keypoints_15
	
def VisualizeDetectedKeypoints(X,Y):
	#Compute mean model
	m= np.mean(Y,axis=0)
	print "m",m.shape
	m= m.reshape(number_of_keypoints,2)
		
	for img in X:
		#print "img",img.shape
		landmarks= stasm.search_single(img)
		landmarks= stasm.force_points_into_image(landmarks, img) #(77,2)
		if len(landmarks)==0:
		#	print 'Detection failed!'
			landmarks= m
		else:
			landmarks= stasm_77_to_kaggle_15(landmarks)
			#don't work
			#landmarks= stasm.search_pinned(landmarks.astype(np.uint8)[:2,:], img)
		
		for point in landmarks:
			img[round(point[1])][round(point[0])] = 255

		cv2.imshow("stasm keypoints", img)
		cv2.waitKey(0)
		
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
	
def Predict(X_test,Y_train):
	#Compute prediction using stasm library
	
	#Compute mean model
	mean_model= np.mean(Y_train,axis=0)
	mean_model= mean_model.reshape(number_of_keypoints,2)
	print "mean_model",mean_model.shape
		
	Y_pred= np.zeros((X_test.shape[0],2*number_of_keypoints),np.float32)
	for i in range(0,X_test.shape[0]):
		print i
		img= X_test[i,:,:]
		#print X_test[i,:,:].shape 
		landmarks= stasm.search_single(img)
		landmarks= stasm.force_points_into_image(landmarks, img) #(77,2)
		if len(landmarks)==0: #Detection failed, so just use mean_model
			landmarks= mean_model
		else:
			landmarks= stasm_77_to_kaggle_15(landmarks)
		#	#don't work
		#	#landmarks= stasm.search_pinned(landmarks.astype(np.uint8)[:2,:], img)
		
		#show prediction
		#img= X_test[i,:,:]
		#for point in landmarks:
		#	img[round(point[1])][round(point[0])] = 255

		#cv2.imshow("stasm keypoints", img)
		#cv2.waitKey(0)
		
		Y_pred[i,:]= landmarks.reshape(1,2*number_of_keypoints)
		
	return Y_pred

#--------------------------------------------------------------------------
X_train,Y_train= LoadTrainData('kaggle_data/training.csv', flDropNaNs= True)
VisualizeDetectedKeypoints(X_train,Y_train)

X_test= LoadTestData('kaggle_data/test.csv')
Y_pred= Predict(X_test,Y_train)
WritePredictionToCsvInKaggleFormat(Y_pred)