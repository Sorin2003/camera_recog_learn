import cv2

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 200)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object list")
print(classes)
# Init camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # get frames
    ret, frame = cap.read()

    # obj detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        class_name = classes[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 20), 3)
        cv2.putText(frame, class_name, (x, y - 8), cv2.FONT_HERSHEY_PLAIN, 2, (200, 255, 0), 2)

    print("class_ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()