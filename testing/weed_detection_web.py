from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os
template_dir = os.path.abspath('C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/testing/templates')
app = Flask(__name__, template_folder=template_dir)

# YOLO Paths
labelsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/obj.names'
weightsPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/weights/crop_weed_detection.weights'
configPath = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/traning/crop_weed.cfg'
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

UPLOAD_FOLDER = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/testing/static/uploads'
OUTPUT_FOLDER = 'C:/Users/vusal/OneDrive/Desktop/Weed-Detection-main/Weed-Detection-main/testing/static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded"
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Process image and save output
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    process_image(filepath, output_path)

    # Pass only filename (not full path) to template
    return render_template('result.html', filename=filename)


def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
    confi = 0.5
    thresh = 0.5
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes, confidences, classIDs = [], [], []
    
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
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    app.run(debug=True)
