import cv2
import numpy as np
import imutils
import time
import os

arguments = {
        "input" : "videos/car_chase.mp4",
        "output" : "output/output.mp4",
        "yolo" : "yolo-coco/",
        "confidence" : 0.5,
        "threshold" : 0.3
        }

labels_path = os.path.sep.join([arguments["yolo"], "coco.names"])

labels = open(labels_path).read().strip().split("\n")

np.random.seed(42)

colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

weights_path = os.path.sep.join([arguments["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([arguments["yolo"], "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

video_stream = cv2.VideoCapture(arguments["input"])

writer = None

while True:
    
    (grabbed, frame) = video_stream.read()
    
    if not grabbed:
        break
    
    (h, w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                 swapRB=True, crop=False)
    
    net.setInput(blob)
    start_time = time.time()
    print("start time : ", start_time)
    layer_outputs = net.forward(layer_names)
    
    end_time = time.time()
    print("Time taken : ", end_time - start_time)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > arguments["confidence"]:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width/2))
                y = int(center_y - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    nms_boxes = cv2.dnn.NMSBoxes(boxes, confidences, arguments["confidence"], 
                                 arguments["threshold"])
    if len(nms_boxes):
        for i in nms_boxes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        color, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MPEG")
        writer = cv2.VideoWriter(arguments["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
        
    writer.write(frame)
        
writer.release()
video_stream.release()
        