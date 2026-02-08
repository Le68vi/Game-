from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib 
import cv2
import winsound
freq=2500
duration=1000

def eyeAspectRatio(eye):
    #vertical
    A=dist.euclidean(eye[1],eye[5])#e=a(x,y)
    B=dist.euclidean(eye[2],eye[4])
    #horizontal
    C=dist.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

count=0
earThresh=0.3#distance between vertical eye coordinate Threshold
earFrames=48#consective frames for eye closure
shapePredictor="shape_predictor_68_face_landmarks.dat"

cam=cv2.VideoCapture(1)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(shapePredictor)

#get the coord of left and right eye
(Istart,IEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    frame=cam.read()
    frame=imutils.resize(frame,width=800)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,0)
    for rect in rects:
        #df is your dataframe
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        lefteye=shape[IStart:IEnd]
        righteye=shape[rStart:rEnd]

        leftEar=eyeAspectRatio(lefteye)
        rightEar=eyeAspectRatio(righteye)

        ear=(leftEar+rightEar)/2.0

        lefteyeHull=cv2.convexHull(lefteye)
        righteyeHull=cv2.convexHull(righteye)
        cv2.drawContours(frame,[lefteyeHull],-1,(0,0,255),1)
        cv2.drawContours(frame,[righteyeHull],1,(0,0,255),1)

        if ear<earThresh:
            count+=1

            if count>=earFrames:
                cv2.putText(frame,"drowsiness Detected",(10,30)
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                winsound.Beep(frequency,duration)
            else:
                count=0
        cv2.imshow("Frame",frame)
        key=cv2.waitKey(1)& 0xFF

        if Key==ord("q"):
            break

cam.release()
cv2.destroyAllWindows()


