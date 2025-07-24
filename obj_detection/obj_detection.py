import cv2
import argparse
import numpy as np

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to input image')
ap.add_argument('-c', '--config', required=True, help='Path to YOLO config file')
ap.add_argument('-w', '--weights', required=True, help='Path to YOLO pre-trained weights')
ap.add_argument('-cl', '--classes', required=True, help='Path to text file containing class names')
ap.add_argument('-o', '--output', default='object-detection.jpg', help='Path to save output image')
args = ap.parse_args()

# Function to get output layers of the YOLO model
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to draw bounding box and label
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Load image
image = cv2.imread(args.image)
Width, Height = image.shape[1], image.shape[0]
scale = 0.00392

# Load class names
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Random colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load YOLO model
net = cv2.dnn.readNet(args.weights, args.config)

# Preprocess image
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))

# Initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

# Process each detection
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Max Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
indices = np.array(indices).flatten()

# Draw bounding boxes
for i in indices:
    x, y, w, h = boxes[i]
    draw_prediction(image, class_ids[i], confidences[i], x, y, x + w, y + h)

# Show and save result
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.imwrite(args.output, image)
cv2.destroyAllWindows()
