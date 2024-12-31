import numpy as np
import cv2
import requests
import random
import logging
from concurrent.futures import ThreadPoolExecutor

# Đọc dữ liệu từ webcam or video
video_path = './test_image/cars.mp4'
cap = cv2.VideoCapture(video_path)

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    line_length = 15

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + line_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + line_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - line_length), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - line_length), color, thickness)

    return image

def send_frame(frame):
    """
    Gửi một frame tới server sử dụng requests (đồng bộ)
    """
    imencoded = cv2.imencode(".jpg", frame)[1]
    files = {'file': ('image.jpg', imencoded.tobytes(), 'image/jpeg')}
    
    try:
        response = requests.post("http://localhost:6000/detections", files=files)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Kết quả từ API: {result}")
            return result
        else:
            logger.error(f"Đã xảy ra lỗi: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return None

def process_frame():
    frame_nmr = -1
    with ThreadPoolExecutor(max_workers=2) as executor:  # Giới hạn số lượng luồng song song
        futures = []
        while True:
            frame_nmr += 1
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            frame = cv2.resize(frame, (1020, 500))

            # Gửi frame song song bằng ThreadPoolExecutor
            futures.append(executor.submit(send_frame, frame))

            # Xử lý kết quả từ các frame đã gửi
            if len(futures) >= 5:
                for future in futures:
                    result = future.result()  # Lấy kết quả từ từng request
                    if result:
                        for prediction in result:
                            bbox = prediction['predictions']['boxes']
                            tracking_id = prediction['predictions']['tracking_ids']
                            confidence = prediction['predictions']['confidence']
                            # Vẽ hình chữ nhật lên khung hình
                            xmin, ymin = bbox[0], bbox[1]
                            thickness = 2  # Độ dày đường viền
                            frame = draw_box(frame, bbox, (colors[int(tracking_id) % len(colors)]), thickness)
                            label = "{}-{}".format(tracking_id, confidence)
                            cv2.putText(frame, str(label), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Hiển thị ảnh kết quả trên webcam
                cv2.imshow("faces", frame)
                
                # Kiểm tra xem có nhấn phím 'q' để thoát không
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                futures = []  # Xoá futures đã xử lý

        # Giải phóng camera và đóng cửa sổ hiển thị
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()
