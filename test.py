import numpy as np
import cv2
from moviepy import editor

def region_selection(image):
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    if lines is None:
        return None,None    
    for line in lines:
        if line is None:
            continue
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    poly_vertices = []
    order = [0,1,3,2]
    line_image = np.zeros_like(image)
    rows, cols = line_image.shape[:2]
    y1 = rows
    y2 = int(rows * 0.6)
    left_line, right_line = lines
    if left_line is not None and right_line is not None:
        left_x1, left_y1 = left_line[0]
        left_x2, left_y2 = left_line[1]
        right_x1, right_y1 = right_line[0]
        right_x2, right_y2 = right_line[1]

        poly_vertices.append((left_x1, left_y1))
        poly_vertices.append((left_x2, left_y2))
        poly_vertices.append((right_x1, right_y1))
        poly_vertices.append((right_x2, right_y2))

        poly_vertices = [poly_vertices[i] for i in order]
        cv2.fillPoly(line_image, pts=[np.array(poly_vertices, 'int32')], color=(0, 0, 255))
        cv2.line(line_image, ((left_x1+right_x1)//2, y1), ((left_x2+right_x2)//2, y2), [0, 255, 0], 5)

    cv2.line(line_image, (cols // 2, y2), (cols // 2, rows), [255, 255, 0], 3)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

def process_camera():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = frame_processor(frame)
        cv2.imshow('Lane Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Lane Detection', cv2.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()

# calling driver function
process_camera()
