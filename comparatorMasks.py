import cv2
import sys
import os
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def ssim(imageA,imageB):
    return measure.compare_ssim(imageA, imageB)


def compareMasks(imageA, imageB):
	mseValue = mse(imageA,imageB)
	#imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	#imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	#print('the ssim is: '+str(ssim(imageA_gray,imageB_gray)))
	print('the MSE is: '+str(mseValue))
	if (mseValue < 9500):
		print('cat recognized :D')
	else:
		print('not found :(')



imageNameA = sys.argv[1]
imageNameB = sys.argv[2]
imageA = cv2.imread('./media/input/testDiferencia/mix/'+imageNameA+'.png',cv2.IMREAD_COLOR)
imageB = cv2.imread('./media/input/testDiferencia/tom/'+imageNameB+'.png',cv2.IMREAD_COLOR)
compareMasks(imageA,imageB)
