# facerec.py
import cv2, sys, numpy, os
from PIL import Image
from six import StringIO
import requests

size = 4

# location of OpenCV Haar Cascade Classifiers:


# xml files describing our haar cascade classifiers
faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "haarcascade_mcs_nose.xml"
 
# build our cv2 Cascade Classifiers
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
 
#-----------------------------------------------------------------------------
#       Load and configure mustache (.png with alpha transparency)
#-----------------------------------------------------------------------------
 
# Load our overlay image: mustache.png
imgMustache = cv2.imread('mustache.png',-1)
 
# Create the mask for the mustache
orig_mask = imgMustache[:,:,3]
 
# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)
 
# Convert mustache image to BGR
# and save the original image size (used later when re-sizing the image)
imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

datasets = 'datasets'

# Part 1: Create fisherRecognizer
print('Training...')
# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use fisherRecognizer on camera stream

webcam = cv2.VideoCapture(0)

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(50, 50),
        flags=0
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1]<500:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = im[y:y+h, x:x+w]
 
            # Detect a nose within the region bounded by each face (the ROI)
            nose = noseCascade.detectMultiScale(roi_gray)
 
            for (nx,ny,nw,nh) in nose:
                # Un-comment the next line for debug (draw box around the nose)
                #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
     
                # The mustache should be three times the width of the nose
                mustacheWidth =  3 * nw
                mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
     
                # Center the mustache on the bottom of the nose
                x1 = nx - (mustacheWidth/4)
                x2 = nx + nw + (mustacheWidth/4)
                y1 = ny + nh - (mustacheHeight/2)
                y2 = ny + nh + (mustacheHeight/2)
     
                # Check for clipping
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > w:
                    x2 = w
                if y2 > h:
                    y2 = h
     
                # Re-calculate the width and height of the mustache image
                mustacheWidth = x2 - x1
                mustacheHeight = y2 - y1
     
                # Re-size the original image and the masks to the mustache sizes
                # calcualted above
                mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
     
                # take ROI for mustache from background equal to size of mustache image
                roi = roi_color[y1:y2, x1:x2]
     
                # roi_bg contains the original image only where the mustache is not
                # in the region that is the size of the mustache.
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
     
                # roi_fg contains the image of the mustache only where the mustache is
                roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
     
                # join the roi_bg and roi_fg
                dst = cv2.add(roi_bg,roi_fg)
     
                # place the joined image, saved to dst back over the original image
                roi_color[y1:y2, x1:x2] = dst
     

            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255))
            crop_img = im[y:y+h, x:x+w]
            cv2.imshow("cropped", crop_img)

            frame_im = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(frame_im)
            stream = StringIO()
            pil_im.save(stream, format="JPEG")
            stream.seek(0)
            img_for_post = stream.read()    
            files = {'image': img_for_post}
            response = requests.post(
                url='/api/path-to-your-endpoint/',
                files=files
            )

    	else:
    	  cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))

    cv2.imshow('OpenCV', im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()