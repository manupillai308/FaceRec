from mtcnn.mtcnn import MTCNN
import cv2
import sys

# box x,y,width, height
def draw_image(image, box):
	ulx = box[0]
	uly = box[1]
	width = box[2]
	height = box[3]
	face = image[uly:uly+height, ulx:ulx+width, :][:]
	new_shape = (face.shape[1]*3, face.shape[0]*3)
	face_resized = cv2.resize(face, new_shape)
	scol = ulx+width+20
	srow = uly-2*height
	image[srow:srow+new_shape[1], scol:scol+new_shape[0],:] = face_resized[:]
	cv2.rectangle(image, (scol-10, srow-10, new_shape[0]+20, new_shape[1]+20), (0, 0, 0), 3)
	return image

detector = MTCNN()

path = input("\nEnter path to video file:")
cap = cv2.VideoCapture(path)
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		elif key & 0xFF == ord('f'):	## forwadding
			print('forwadding..')
			while(cv2.waitKey(1) & 0xFF != ord('s')):
				cv2.imshow('Output', cv2.resize(cap.read()[1], (1280,720)))
			continue
		prediction = detector.detect_faces(frame[...,[2,1,0]])
		image = frame[:]
		for i, faces in enumerate(prediction):
			box = faces['box']
			image = draw_image(image, box)
		cv2.imshow('Output',cv2.resize(image, (1280, 720)))
	else:
		break

cap.release()
cv2.destroyAllWindows()
