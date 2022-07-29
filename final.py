from __future__ import print_function
import cv2
import numpy
import math
import os
from pyagilis import agPort, controller, channel, mothreading

MAX_FEATURES = 15000
GOOD_MATCH_PERCENT = 0.1

def calibrate(cam=0):

  # Takes calibration images and finds homography
  refCapture(cam)
  input("Move 100 microns for calibration")
  im, imRef = testCapture(cam)
  h = alignImages(im, imRef)

  # Determines rotation matrix
  movement = [-h[0][2],h[1][2]]
  theta = math.atan2(movement[0], movement[1])

  # Populates rotation matrix
  R = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]

  # Determines conversion factor
  movement_prime = numpy.matmul(R, movement)
  conversion = 100/movement_prime[1]

  input("Calibration completed")

  return R, conversion

def calcDist(h, R, conv):

  # Find movement vector in camera frame
  x = -h[0][2]
  y = h[1][2]
  movement = [x,y]

  # Calculate movement in stage frame
  movement_prime = numpy.matmul(R, movement)
  x_prime = movement_prime[0]
  y_prime = movement_prime[1]

  # Find travel distance in stage frame
  tx = x_prime * conv
  ty = y_prime * conv
  print("x-translation: ", tx)
  print("y-translation: ", ty)

  return tx, ty

def rename(ref, test, i):
  
  # Rename images for next iteration
  cv2.imwrite("ref.jpg", test)
  cv2.imwrite("pic_" + str(i) + ".jpg", ref)

def imgRemove():

  # Remove old ref and test images
  os.remove("ref.jpg")
  os.remove("test.jpg")

def refCapture(port):

  # Take reference image
  cam = cv2.VideoCapture(port)
  result, img = cam.read()
  cv2.imwrite("ref.jpg", img)

def testCapture(port):

  # Take test image
  cam = cv2.VideoCapture(port)
  result, img = cam.read()
  cv2.imwrite("test.jpg", img)

  # Read reference image
  refFilename = "ref.jpg"
  print("Reading reference image : ", refFilename)
  imRef = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = "test.jpg"
  print("Reading image to align : ", imFilename);
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

  return im, imRef

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches = sorted(matches, key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = numpy.zeros((len(matches), 2), dtype=numpy.float32)
  points2 = numpy.zeros((len(matches), 2), dtype=numpy.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  # Write aligned image to disk.
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename);
  cv2.imwrite(outFilename, im1Reg)

  print("Estimated homography : \n",  h)

  return h

def testAlign(R, conv, cam=0, numMovements=1):

  refCapture(cam)

  i=0
  while(i < numMovements):

    # Align images and iterate multiple times
    input("Press Enter to continue")
    im, imRef = testCapture(cam)
    h = alignImages(im, imRef)
    calcDist(h, R, conv)
    rename(imRef, im, i)

    # Iterate variable
    i = i+1

  cv2.imwrite("pic_" + str(i) + ".jpg", imRef)  
  imgRemove()

def run(qx, qy, R, conv, cam=0, error=.5, port="COM7"):

  agl = controller.AGUC2(port)

  refCapture(cam)

  xdif, ydif = qx, qy
  i=0
  while(xdif, ydif > error):

    # Move Agilis stage by the specified distance (microns)
    agl.move(int(xdif/int(agl.axis["X"].stepAmp)), int(ydif/int(agl.axis["Y"].stepAmp)))

    # Align images and iterate multiple times
    im, imRef = testCapture(cam)
    h = alignImages(im, imRef)
    x,y = calcDist(h, R, conv)
    cv2.imwrite("pic_" + str(i) + ".jpg", im)

    # Update differences
    xdif = qx - x 
    ydif = qy - y

    # Iterate variable
    i = i+1

  cv2.imwrite("pic_" + str(i) + ".jpg")  
  imgRemove()
