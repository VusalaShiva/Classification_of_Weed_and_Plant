import cv2
import numpy as np
import time
import os

# Loading class labels YOLO model was trained on
labelsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/obj.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Load weights and cfg
weightsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/weights/crop_weed_detection.weights'
configPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/crop_weed.cfg'

# Color selection for drawing bounding boxes
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Load input image and get its dimensions
image_path = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/testing/images/crop_3.jpeg'
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# Parameters
confi = 0.5
thresh = 0.5

# Get YOLO output layer names
ln = net.getLayerNames()
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # Old format
except:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]  # New format

# Construct a blob from the input image and perform a forward pass
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# Initialize detection lists
boxes = []
confidences = []
classIDs = []

# Process detections
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confi:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# Apply non-maxima suppression
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

print("[INFO] Detections done, drawing bounding boxes...")
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        print(f"[OUTPUT]: detected label -> {LABELS[classIDs[i]]}")
        print(f"[ACCURACY]: {confidences[i]:.4f}")

# Convert image to RGB and display
cv2.imshow('Output', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()

print("[STATUS]: Completed")
