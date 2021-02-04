# facerec.py
import cv2, sys, numpy, os, time
import face_recognition
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'

# Part 1: Create LBPHRecognizer
print('Training...')

# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id=id+1
(im_width, im_height) = (92, 72) 

#Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]

#OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)

# Part 2: Use LBPHRecognizer on camera stream
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)


while True:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)                                                         
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    for face in faces:
        face_i = face
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        print(prediction,names[prediction[0]])
        if prediction[1]<=110 and prediction[1]>=60:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(im,(names[prediction[0]]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0),2)
            det = face_recognition.api.face_landmarks(im)
            for de in det:    
                for i in de.keys():
                    for y in range(len(de[i])):
                        cv2.circle(im,de[i][y],1,(0, 255, 0),-1)
        else:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow('..Facial Recogonision..', im)
    if cv2.waitKey(5) & 0xFF == ord('q'):
            break

