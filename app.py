from sort import *
import cv2, numpy as np, pandas as pd
from Yolov8.yolov8 import YOLOv8_det # Import YOLOv8 object detector
import loguru, argparse, random
from utils import draw_box, get_class

# Argument parser
args = argparse.ArgumentParser()
args.add_argument('--model', type=str, help='Path the model', default='./models/static_quantized.onnx')
args.add_argument('--video', type=str, help='Path the video', default='./test_image/cars.mp4')
args.add_argument('--output', type=str, help='Path to save in the video output', default='./test_image/cars_out.mp4')
args.add_argument('--vehicle_class', type=list, help='detection class', default=[2, 3, 5, 7])
args = args.parse_args()

# Set of class IDs for vehicles
vehicles = set(args.vehicle_class) 

# Load video
cap = cv2.VideoCapture(args.video)

# cap_out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),
#                           (1020, 500))

# Initialize YOLOv8 object detector
my_yolov8 = YOLOv8_det(args.model)

# Object tracking
mot_tracker = Sort()

class_names = get_class()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

def main():
    loguru.logger.info("Starting the application")
    frame_nmr = -1
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            frame = cv2.resize(frame, (1020, 500))
            detections = my_yolov8.detect_objects(frame)
            boxes, scores, class_ids = detections
            detections_ = []
            # frame = my_yolov8.draw_detections(frame)
            for i in range(len(boxes)):
                box = boxes[i].tolist()
                score = scores[i].tolist()
                class_id = class_ids[i].tolist()
                if int(class_id) in vehicles:
                    detections_.append(box + [score, class_id])
            
            # px = pd.DataFrame({
            #         'x1': [x[0] for x in detections_],
            #         'y1': [x[1] for x in detections_],
            #         'x2': [x[2] for x in detections_],
            #         'y2': [x[3] for x in detections_],
            #         'score': [x[4] for x in detections_],
            #         'class_id': [x[5] for x in detections_]
            #     }).astype("float")        
            # print(px)
            if detections_:
                track_ids = mot_tracker.update(np.asarray(detections_))
                for track_id in track_ids:
                    # print(dir(track_id))
                    # loguru.logger.info("Track ID: {}".format(track_id))
                    x3, y3, x4, y4, id = map(int, track_id)
                    # print(x3, " ", y3, " ", x4, " ", y4, " ", id)
                    bbox = [x3, y3, x4, y4]
                    thickness = 2
                    label = "ID:{}".format(id)
                    frame = draw_box(frame, label, np.array(bbox), (colors[int(id) % len(colors)]), thickness)

                # cap_out.write(frame)
                cv2.imshow("frames", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    # cap_out.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()