import torch
import numpy as np
import cv2
from MTCNN.mtcnn_opencv import MTCNN
from facerecognition.Learner import face_learner
import glob
from tqdm import tqdm

detect = MTCNN()
recognise = face_learner()
names = []
embeds = []
cnt = 0
img_path = 'embed_images'
embedding_name = 'checkpoints/facebank/edge_embed.npy'
ID_names = 'checkpoints/facebank/edge.npy'


for item in tqdm(glob.glob(f'{img_path}/*.jpg')):
    img = cv2.imread(item)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox,faces,landmarks = detect.get_align_multiface(img)
    img = cv2.resize(faces[0],(112,112))
    embed = recognise.get_embeddings([img])
    #saving
    person = (item.split('/')[-1]).split('.')[0]
    names.append(person)
    embeds.append(embed)
    cnt += 1


print('total people:',cnt)
print(len(embeds))
print(names)

embedding = np.asarray(embeds)
embedding = embedding.transpose(0,2,1).squeeze(-1)
print('final shape:',embedding.shape , type(embedding))
np.save(embedding_name, embedding)

name_list = np.asarray(names)
np.save(ID_names, names)

print('done')
