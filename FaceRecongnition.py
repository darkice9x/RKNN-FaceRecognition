import get_embedding
import cv2
import sys
import numpy as np
import os
import time

def cosine_similarity1(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

def extract_filename(file_path):
    full_filename = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(full_filename)[0]
    return filename_without_extension

if __name__ == "__main__":
    image_org = "./img/xu_1.jpg"
    #image4 = "./face.jpg"
    images = [ "./img/select_sy2_face.jpg", "./img/select_sy2_face.jpg", "./img/lin_1.jpg", "./img/lin_2.jpg", "./img/lin_3.jpg", "./img/xu_2.jpg", "./img/xu_3.jpg"]
    path=os.getcwd()

    get_embedding.init()
    img1 = cv2.imread(image_org)
    start_time = time.time()
    get_face1 = get_embedding.get_embeddings(img1)
    feature1 = get_face1[0]['embedding']
    print( f'get time : {(time.time() - start_time)*1000}ms')

    detect_list = []
    for j in range(len(images)):
        img2 = cv2.imread(images[j])
        start_time = time.time()
        get_face2 = get_embedding.get_embeddings(img2)
        personname = extract_filename(images[j])
        for i in range(len(get_face2)):
            feature2 = get_face2[i]['embedding']
            #print( f'get time : {(time.time() - start_time)*1000}ms')
            match, cosine_similarity = get_embedding.compare_face(feature1, feature2)
            #cosine_similarity_1 = cosine_similarity1(feature1[0], feature2[0])
            #print( f'name: {personname}, match: {match}, face_distances : {cosine_similarity[0]}, {cosine_similarity_1}'  )
            print( f'name: {personname}, match: {match}, face_distances : {cosine_similarity[0]}'  )
            if match == [True]:
                detect_list.append([j,i,cosine_similarity[0] ])

    if len(detect_list)>0 :
        max_value = max(detect_list, key=lambda x: x[2])[0]
        file_path = images[max_value]
        # Extract only the file name
        personname = extract_filename(file_path)
        print("Person name :", personname)
    else:
        print( "Unkown!!")