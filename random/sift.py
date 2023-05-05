import cv2

# Load the image
image = cv2.imread("../images/dota2.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create the SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors using the SIFT algorithm
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# Draw the detected keypoints on the image
image_with_keypoints = cv2.drawKeypoints(
    gray_image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Show the image with detected keypoints
cv2.imshow("SIFT Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
