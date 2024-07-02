import cv2
import numpy as np

def detect_lanes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI)
    height, width = image.shape[:2]
    mask = np.zeros_like(edges)
    roi = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    # Draw detected lines on the original image
    lane_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine the original image with the detected lines
    result = cv2.addWeighted(image, 0.8, lane_image, 1, 0)
    
    return result

# Test the function on a sample image
image = cv2.imread('sample_image.jpg')
result = detect_lanes(image)
cv2.imshow('Lane Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()