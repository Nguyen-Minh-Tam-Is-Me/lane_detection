import numpy as np
import cv2
from moviepy import editor

def region_selection(image):
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

def fit_polynomial(image, lines):
    if lines is None:
        return None, None
    left_points = []
    right_points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_points.extend([(x1, y1), (x2, y2)])
            else:
                right_points.extend([(x1, y1), (x2, y2)])
    if left_points:
        left_points = np.array(left_points)
        left_poly = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
    else:
        left_poly = None
    if right_points:
        right_points = np.array(right_points)
        right_poly = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
    else:
        right_poly = None
    return left_poly, right_poly

def draw_lane_lines(image, left_poly, right_poly, color=[255, 0, 0], thickness=12):
    line_image = np.zeros_like(image)
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if left_poly is not None:
        left_x1 = int(np.polyval(left_poly, y1))
        left_x2 = int(np.polyval(left_poly, y2))
        cv2.line(line_image, (left_x1, y1), (left_x2, y2), color, thickness)
    if right_poly is not None:
        right_x1 = int(np.polyval(right_poly, y1))
        right_x2 = int(np.polyval(right_poly, y2))
        cv2.line(line_image, (right_x1, y1), (right_x2, y2), color, thickness)
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
    left_poly, right_poly = fit_polynomial(image, hough)
    result = draw_lane_lines(image, left_poly, right_poly)
    return result

def process_video(test_video, output_video):
    input_video = editor.VideoFileClip(test_video, audio=False)
    processed = input_video.fl_image(frame_processor)
    processed.write_videofile(output_video, audio=False)

process_video('solidYellowLeft.mp4', 'output.mp4')
