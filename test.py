import cv2
from pyzbar.pyzbar import decode

# Load the image
image = cv2.imread('/Users/hash/Downloads/yUXvD.jpg')
image_height, image_width, _ = image.shape  # Get image dimensions

# Detect and decode the QR code
decoded_objects = decode(image)

# Process the detected QR codes
for obj in decoded_objects:
    print("Type:", obj.type)
    print("Data:", obj.data.decode("utf-8"))

    # Get the bounding box points
    points = obj.polygon

    if len(points) == 4:  # If the QR code is a rectangle
        # Extract x and y values from the points
        x_values = [point.x for point in points]
        y_values = [point.y for point in points]

        # Calculate Left, Top, Width, and Height in pixels
        left = min(x_values)
        top = min(y_values)
        width = max(x_values) - left
        height = max(y_values) - top

        # Normalize the bounding box values
        norm_left = left / image_width
        norm_top = top / image_height
        norm_width = width / image_width
        norm_height = height / image_height

        # Print the normalized bounding box
        print(f"Normalized Bounding Box - Left: {norm_left}, Top: {norm_top}, Width: {norm_width}, Height: {norm_height}")

        # Optionally, draw the rectangle (in pixel values) on the image
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

# Display the image with the drawn rectangle (if needed)
cv2.imshow("QR Code Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()