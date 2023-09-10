import cv2
import time 
import argparse
import sys
sys.path.append('/home/rubayet/miniconda3/envs/gr/lib/python3.8/site-packages')
from MTCNN.mtcnn_opencv import MTCNN
from facerecognition.Learner import face_learner
from config import conf

def fr_pipeline(conf, args, faceAPI, detect_faces):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        isSuccess,frame = cap.read()
        try:
            if isSuccess:
                fb_name = conf['facebank']['client'][0]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox, cropped_face, _ = detect_faces.get_align_multiface(frame)
                user_id, score = faceAPI.infer(cropped_face,fb_name)
                if user_id == None:
                    continue
                # if bbox != None:
                #     for i,box in enumerate(bbox):
                #         upper_left = (box[0],box[1])
                #         lower_right = (box[0]+box[2], box[1]+box[3])
                #         cv2.rectangle(frame,upper_left, lower_right, (0, 155, 255), 2)
                #         cv2.putText(frame, str(user_id[i]),(upper_left), cv2.FONT_HERSHEY_SIMPLEX, 1,(124,252,0),3,cv2.LINE_AA)
                print(user_id)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # cv2.imshow("Normal Read", frame)
        except Exception as e:
            print(e)
    cap.release()

def start(conf, args):
    faceAPI = face_learner()
    detect_faces = MTCNN()
    fr_pipeline(conf, args, faceAPI, detect_faces)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--show",  action="store_true", help="show the faces in this device?")
	parser.add_argument("-v", "--verbose", action="store_true", help="print the response")
	parser.add_argument("-a", "--accessway", type=str, help="it sets in and out camera")
	args = parser.parse_args()
	conf = conf.get_configurataion()
	start(conf, args)