import cv2
import os
import numpy as np

labelsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/obj.names'
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/weights/crop_weed_detection.weights'
configPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/crop_weed.cfg'

# Color selection for drawing bbox
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Parameters
confi = 0.5
thresh = 0.5
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# If you want to use webcam instead of video, uncomment below line and comment line 24
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/testing/crop_weed_video.mp4")

while True:
    ret, image = cap.read()
    if ret:
        (H, W) = image.shape[:2]
        
        # Construct a blob from the input image and then perform a forward pass of the YOLO object detector
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        
        boxes = []
        confidences = []
        classIDs = []
        
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
        
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                print("Predicted ->  :  ", LABELS[classIDs[i]])
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
        cv2.imshow('frame', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()