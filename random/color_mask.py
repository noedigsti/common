import cv2 as cv
import numpy as np


def preprocess_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        print("Image not found or unable to read.")
        return

    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply a Gaussian blur
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # Use Canny edge detection
    edges = cv.Canny(blurred_img, 100, 200)

    # Dilate the edges to close gaps
    dilated_edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    return img, dilated_edges


def find_contours(dilated_edges):
    # Find contours
    contours, _ = cv.findContours(
        dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # Filter contours based on area
    min_area = 1000
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

    return filtered_contours


def create_mask(img, filtered_contours):
    # Create a mask for the contours
    mask = np.zeros_like(img)
    cv.drawContours(mask, filtered_contours, -1, (255, 255, 255), -1)

    return mask


def apply_mask(img, mask):
    # Apply the mask to the original image
    separated_objects = cv.bitwise_and(img, mask)

    return separated_objects


def display_result(img, separated_objects):
    cv.imshow("Original Image", img)
    cv.imshow("Separated Objects", separated_objects)
    cv.waitKey(0)
    cv.destroyAllWindows()


image_path = "../images/dota2.jpg"
img, dilated_edges = preprocess_image(image_path)
filtered_contours = find_contours(dilated_edges)
mask = create_mask(img, filtered_contours)
separated_objects = apply_mask(img, mask)
display_result(img, separated_objects)
