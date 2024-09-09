import boto3

# Initialize the Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')

# Create a collection to store face data (if not already created)
collection_id = "my_faces_collection"
rekognition.create_collection(CollectionId=collection_id)

def enroll_face(image_path, collection_id):
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    
    # Index faces in the collection
    response = rekognition.index_faces(
        CollectionId=collection_id,
        Image={'Bytes': image_bytes},
        ExternalImageId="JohnDoe",  # Unique ID for the user
        DetectionAttributes=['ALL']
    )

    print("Face enrolled successfully:", response)

# Enroll a face
enroll_face("/Users/hash/work/satschel/search_by_face_poc/images/enrollment.png", collection_id)
