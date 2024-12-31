import numpy as np, cv2

def draw_box(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    line_length = 15
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0007
    text_thickness = int(min([img_height, img_width]) * 0.002)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

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

    cv2.rectangle(image, (x1, y1-3),
                  (x1 + tw, y1 - th-5), color, -1)

    cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return image

def get_class():
    path = "coco.txt"
    my_file = open(path, "r")
    data = my_file.read()
    class_names = data.split("\n")
    return class_names