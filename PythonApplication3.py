# To calibrate, set larger QR code on top of Agilis QR code and place camera over top
# Move stage by 100 microns rotating dial to the right (CCW/to the larger number)
# Move larger QR code out of the way without touching camera and continue the program

# If the program is not working many times in a row, most likely it is at its limit
# If this happens change the distance to move to the opposite sign

# Import statements
import sys
import clr
from time import sleep
import cv2
import numpy
import os
import math
from decimal import Decimal

# Appends the path needed for Agilis command library and imports
sys.path.append(r"C:\Program Files\Newport\Piezo Motion Control\Newport AG-UC2-UC8 Applet\Bin")
clr.AddReference('AgilisCmdLib')
from Newport.AgilisCmdLib import *

# Class to control the Agilis stage
class Agilis:

    def __init__(self, stepAmp):

        # Creates Agilis Command Library and opens port
        self.agl = AgilisCmds()
        self.DevicePorts = self.agl.GetDevices()
        self.agl.OpenInstrument("COM7")
        self.agx, self.agy = int(1), int(2)
        self.err = ''

        # Sets the controller to remote mode and sets step amplitude
        self.agl.MR(self.err)
        self.setStepAmp(stepAmp)
        print("Controller Ready")


    def amIstill(self, axis):

        resp = int(1)

        # Checks that the stage has stopped moving
        while(resp != (0,0,'')):

            resp = self.agl.TS(axis, 1, self.err)
            sleep(.1)


    def move(self, x, y):

        # Moves each stage by specified amount
        self.agl.PR(self.agx, x, self.err)
        self.amIstill(self.agx)
        self.agl.PR(self.agy, y, self.err)
        self.amIstill(self.agy)

    
    def stepAmp(self, axis):

        # Returns the step amplitude of an axis
        posamp = self.agl.SU_Get(axis, '+', 0, self.err)
        negamp = self.agl.SU_Get(axis, '-', 0, self.err)

        return posamp[1]


    def setStepAmp(self, stepAmp):

        # Sets the step amplitude for a given axis
        self.agl.SU_Set(self.agx, '+', stepAmp, self.err)
        self.agl.SU_Set(self.agx, '-', stepAmp, self.err)
        self.agl.SU_Set(self.agy, '+', stepAmp, self.err)
        self.agl.SU_Set(self.agy, '-', stepAmp, self.err)


# Class to control the vibrometer setup
class preAligner:

    def __init__(self, stepAmp=50):

        # Defines max number of matches and creates Agilis object
        self.MAX_FEATURES = 10000
        self.GOOD_MATCH_PERCENT = 0.08
        self.UC2 = Agilis(stepAmp)
      

    def calibrate(self, cam=0):

        # Takes calibration images and finds homography
        self.refCapture(cam)
        input("Move 100 microns for calibration")
        im, imRef = self.testCapture(cam)
        h = self.alignImages(im, imRef)

        # Determines rotation matrix
        movement = [-h[0][2],h[1][2]]
        theta = math.atan2(movement[0], movement[1])

        # Populates rotation matrix
        R = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]

        # Determines conversion factor
        movement_prime = numpy.matmul(R, movement)
        conv = 100/movement_prime[1]
        print(conv)

        input("Calibration completed")
        self.imgRemove()

        return R, conv


    def calcDist(self, h, R, conv):

        # Find movement vector in camera frame
        x = -h[0][2]
        y = h[1][2]
        movement = [x,y]

        # Rotate vector using rotation matrix
        movement_prime = numpy.matmul(R, movement)
        x_prime = movement_prime[0]
        y_prime = movement_prime[1]

        # Find travel distance in stage frame
        tx = x_prime * conv
        ty = y_prime * conv
        print("x-translation: ", tx)
        print("y-translation: ", ty)

        return tx, ty


    def rename(self, ref, test, i):
  
        # Rename images for next iteration
        cv2.imwrite("ref.jpg", test)
        cv2.imwrite("pic_" + str(i) + ".jpg", ref)


    def imgRemove(self):

        # Remove old ref and test images
        os.remove("ref.jpg")
        os.remove("test.jpg")


    def refCapture(self, port):

        # Take reference image
        cam = cv2.VideoCapture(port)
        result, img = cam.read()
        cv2.imwrite("ref.jpg", img)


    def testCapture(self, port):

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


    def alignImages(self, im1, im2):

        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
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

        return h


    def testAlign(self, R, conv, numMovements=1, cam=0):

        self.refCapture(cam)

        i=0
        while(i < numMovements):

            # Align images and iterate multiple times
            input("Press Enter to continue")
            im, imRef = self.testCapture(cam)
            h = self.alignImages(im, imRef)
            self.calcDist(h, R, conv)
            self.rename(imRef, im, i)

            # Iterate variable
            i = i+1

        cv2.imwrite("pic_" + str(i) + ".jpg", imRef)
        self.imgRemove()


    def run(self, R, conv, qx, qy, error=1, cam=0):

        self.refCapture(cam)

        xdif, ydif = qx, qy
        i=0
        while(abs(xdif) >= error or abs(ydif) >= error):

            # Move Agilis stage by the specified distance (microns)
            self.UC2.move(int(xdif), int(ydif))

            # Align images and iterate multiple times
            im, imRef = self.testCapture(cam)
            h = self.alignImages(im, imRef)

            # Calculates distance using correct method
            x,y = self.calcDist(h, R, conv)

            # Update differences (ignores errors in rotation)
            if(int(xdif) != 0):
                xdif = qx - x
            if(int(ydif) != 0):
                ydif = qy - y
            print(xdif, ydif)

            i = i+1
        
        cv2.imwrite("translated_" + str(i) + ".jpg", im)
        cv2.imwrite("reference_" + str(i) + ".jpg", imRef)
        self.imgRemove()



if __name__ == '__main__':

    # Instantiates vibrometer class
    PA = preAligner()

    R, conv = PA.calibrate()
    PA.run(R, conv, -100, 100)
        
