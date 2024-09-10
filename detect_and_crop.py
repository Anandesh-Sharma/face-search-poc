
import boto3
from PIL import Image
import io

# Initialize AWS Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')

def detect_faces(image_bytes):
    """
    Detect faces in an image using AWS Rekognition.
    
    :param image_bytes: The image data in bytes.
    :return: List of face details including bounding boxes.
    """
    response = rekognition.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']
    )
    return response['FaceDetails']

def crop_faces(image_bytes, face_details):
    """
    Crop faces from an image using bounding box data.
    
    :param image_bytes: The image data in bytes.
    :param face_details: Detected face details from Rekognition.
    :return: List of cropped face images as PIL Image objects.
    """
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    face_images = []

    # Crop each detected face
    for face in face_details:
        box = face['BoundingBox']
        left = int(box['Left'] * width)
        top = int(box['Top'] * height)
        right = int(left + (box['Width'] * width))
        bottom = int(top + (box['Height'] * height))

        # Crop the face and store it in memory
        face_image = img.crop((left, top, right, bottom))
        face_images.append(face_image)

    return face_images

def search_faces_by_image(face_image):
    """
    Search for a cropped face in a Rekognition collection.
    
    :param face_image: The cropped face as a PIL Image object.
    :return: List of matched faces from the Rekognition collection.
    """
    # Convert the cropped face image to bytes
    face_bytes = io.BytesIO()
    face_image.save(face_bytes, format='JPEG')
    face_bytes = face_bytes.getvalue()

    # Search the face in the Rekognition collection
    response = rekognition.search_faces_by_image(
        CollectionId='satschel-nonprod-public-v1',
        Image={'Bytes': face_bytes},
        MaxFaces=10,
        FaceMatchThreshold=85
    )

    return response['FaceMatches']

def main(image_ndarray):
    """
    Main function to detect, crop, and search faces in an image.
    
    :param image_ndarray: The input image in ndarray format.
    """
    # Convert ndarray to bytes
    image = Image.fromarray(image_ndarray)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes = image_bytes.getvalue()

    # Detect faces in the image
    face_details = detect_faces(image_bytes)

    # Crop the faces using detected face details
    cropped_faces = crop_faces(image_bytes, face_details)

    # Search for each cropped face in the Rekognition collection
    for i, face_image in enumerate(cropped_faces):
        matches = search_faces_by_image(face_image)
        print(f"Face {i+1}: {matches}")


import cv2
image_ndarray = cv2.imread('/Users/hash/work/satschel/search_by_face_poc/images/two.jpeg')
main(image_ndarray)
# # Example usage: pass the image as bytes directly
# with open('/Users/hash/work/satschel/search_by_face_poc/images/two.jpeg', 'rb') as image_file:
#     image_data = image_file.read()

# main(image_data)
