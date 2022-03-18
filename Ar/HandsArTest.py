#!/usr/bin/env python
# coding: utf-8

# In[1]:
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer  

import time, threading
import argparse
import time
import cv2
import numpy as np
import math
import os
import mediapipe as mp
from numpy import interp
import uuid
from typing import Mapping, Tuple
from mediapipe.python.solutions import drawing_styles
import pygame
from OpenGL.GL import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 15
DEFAULT_COLOR = (0, 255, 0)
# load the reference surface that will be searched in the video stream
dir_name = os.getcwd()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


# In[31]:


def MTL(filename):
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError("mtl file doesn't start with newmtl stmt")
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            surf = pygame.image.load("/".join(list(filename.split('/')[0:-1]))+"/"+mtl['map_Kd'])
            image = pygame.image.tostring(surf, 'RGBA', 1)
            ix, iy = surf.get_rect().size
            texid = mtl['texture_Kd'] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, image)
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents


class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        glBegin(GL_POLYGON)
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(filename.replace(".obj",".mtl"))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
#                 self.faces.append((face, norms, texcoords))


#         for face in self.faces:
#             vertices, normals, texture_coords, material = face

#             mtl = self.mtl[material]
#             if 'texture_Kd' in mtl:
#                 # use diffuse texmap
#                 glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
#             else:
#                 # just use diffuse colour
#                 glColor(*mtl['Kd'])

#             glBegin(GL_POLYGON)
#             for i in range(len(vertices)):
#                 if normals[i] > 0:
#                     glNormal3fv(self.normals[normals[i] - 1])
#                 if texture_coords[i] > 0:
#                     glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
#                 glVertex3fv(self.vertices[vertices[i] - 1])
#             glEnd()
#         glDisable(GL_TEXTURE_2D)
#         glEndList()


# In[32]:

@jit(target ="cuda")  
def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img
	
@jit(target ="cuda")  
def renderObj(img, obj, projection, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = (644,372)
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            print("face[-1] : ",face[-1])
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)        

    return img


def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    print(hex_color)
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(200)
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


# In[33]:


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
def getColor(zDist):
    c = int(interp(zDist, [0,15], [0,255]))
    return (c,c,c)


def createLandMarks(hand_landmarks): #-> Mapping[int, mp_drawing.DrawingSpec]:
  hand_landmark_style = {}  
  for k, v in drawing_styles._HAND_LANDMARK_STYLE.items():
    for landmark in k:
      c = getColor(abs(hand_landmarks.landmark[landmark].z*100))
      r = int(abs(hand_landmarks.landmark[landmark].z*100))
      hand_landmark_style[landmark] =   mp_drawing.DrawingSpec(color=c, thickness=drawing_styles._THICKNESS_DOT, circle_radius= r )
  return hand_landmark_style        


# In[ ]:


cap = cv2.VideoCapture(0)
# Load 3D model from OBJ file
obj = OBJ(os.path.join(dir_name, 'models/IronMan/IronMan.obj'), swapyz=True)  
projection = np.float32([[     503.33,   -699.16,    503.33,-130131.43],
                         [    1500,    -62.98,     40.02,-391977.22],
                         [      0.26,      0.22,      0.94,  -1283.31]])
camera_parameters = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]])

homography =  np.float32([[0.4160569997384721, -1.306889006892538, 553.7055461075881],
                          [0.7917584252773352, -0.06341244158456338, -108.2770029401219],
                          [0.0005926357240956578, -0.001020651672127799, 1]])
createControls = 1
counter = 0

def on_change(value):
    valuelf = value/360
    print(valuelf)
    homography[1][1] = valuelf    
    
	
	
def mainFun():
	with mp_hands.Hands(static_image_mode=False,min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
		while cap.isOpened():
			ret, frame = cap.read()
			
			# BGR 2 RGB
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			# Flip on horizontal
			image = cv2.flip(image, 1)
			
			# Set flag
			image.flags.writeable = False
			
			# Detections
			results = hands.process(image)
			
			# Set flag to true
			image.flags.writeable = True
			
			# RGB 2 BGR
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			
	#         print('Handedness:', results.multi_handedness)

			#Get image H ,W
			image_height, image_width, _ = image.shape
			
			# Rendering results
			if results.multi_hand_landmarks:
				for num, hand_landmarks  in enumerate(results.multi_hand_landmarks):

	#                 print(
	#                     f'Index finger tip coordinates: (',
	#                     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
	#                     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height}) '
	#                     f'{abs(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z*100)})'
	#                 )
					mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
	#                                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
	#                                         mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)
										  createLandMarks(hand_landmarks),
										  mp_drawing_styles.get_default_hand_connections_style())                        
									
					lnd1 = hand_landmarks.landmark[4]
					lnd2 = hand_landmarks.landmark[0]
					lnd3 = hand_landmarks.landmark[17]
					lnd4 = hand_landmarks.landmark[8]
					lndLst = np.array([[lnd1.x*image_width, lnd1.y* image_height],
									  [lnd2.x*image_width, lnd2.y* image_height],
									  [lnd3.x*image_width, lnd3.y* image_height], 
									  [lnd4.x*image_width, lnd4.y* image_height],
									  [lnd1.x*image_width, lnd1.y* image_height]]).reshape((-1, 1, 2))
					
					image = cv2.polylines(image, [np.int32(lndLst)], True, 255, 3, cv2.LINE_AA)
					
					src_pts = np.float32([0 , 0 ,
										  500 , 0,
										  500, 500,
										  0, 500]).reshape(-1, 1, 2)
					dst_pts = np.float32([lnd1.x*image_width, lnd1.y* image_height,
										  lnd2.x*image_width, lnd2.y* image_height,
										  lnd3.x*image_width, lnd3.y* image_height,
										  lnd4.x*image_width, lnd4.y* image_height]).reshape(-1, 1, 2) 
					dst_pts = dst_pts.round(2)
					homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
					projection = projection_matrix(camera_parameters, homography)  

					image = renderObj(image, obj, projection, False)

			
			plot = np.zeros([image_height, image_width, 3], dtype=np.uint8)                
			if results.multi_hand_world_landmarks:
				for num,hand_world_landmarks in enumerate(results.multi_hand_world_landmarks):                
					for idx,landMrk in enumerate(hand_world_landmarks.landmark):
						hand_world_landmarks.landmark[idx].x += 0.5
						hand_world_landmarks.landmark[idx].y += 0.5
					mp_drawing.draw_landmarks(plot,hand_world_landmarks, mp_hands.HAND_CONNECTIONS)
	#                 mp_drawing.plot_landmarks(hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
			
			cv2.imshow('Plot', plot)
			cv2.imshow('HandTracking', image) 

			if(createControls):
				createControls = 0
				cv2.createTrackbar('slider', "HandTracking", -100,100, on_change)
			
			
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break

	cap.release()
	cv2.destroyAllWindows()
	
mainFun()