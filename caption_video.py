
import numpy as np
import imutils
import time
import cv2
import requests
import copy
# load the COCO class labels our YOLO model was trained on
labelsPath = "yolo-coco\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("created_video.mp4")
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
count = -1
while True:

	if True :
		start_time = time.time()
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		count += 1
		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break
		if 	count%30 == 0 :
			# if the frame dimensions are empty, grab them
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# construct a blob from the input frame and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes
			# and associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			start = time.time()
			layerOutputs = net.forward(ln)
			end = time.time()

			# initialize our lists of detected bounding boxes, confidences,
			# and class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []

			original_frame = copy.deepcopy(frame)

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability)
					# of the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5:
						# scale the bounding box coordinates back relative to
						# the size of the image, keeping in mind that YOLO
						# actually returns the center (x, y)-coordinates of
						# the bounding box followed by the boxes' width and
						# height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top
						# and and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates,
						# confidences, and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# apply non-maxima suppression to suppress weak, overlapping
			# bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
				0.5)
			send_request = False
			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bicycle" or LABELS[classIDs[i]] == "motorbike":
						print("Vehicle detected")
						send_request = True
			both = np.hstack((original_frame, frame))

			cv2.imshow("window", both)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

			if send_request:
				cv2.imwrite("detected.jpg",original_frame)
				url = "http://localhost:5000/generate"

				with open('detected.jpg',"rb") as f:
					image_to_be_sent = f.read()
				files = {
					"file": image_to_be_sent
				}
				payload = {
					"class_name" : "general"
				}

				response = requests.request("POST", url, files=files, data=payload)

				print(response.text)


			print(count)




print("[INFO] cleaning up...")
vs.release()