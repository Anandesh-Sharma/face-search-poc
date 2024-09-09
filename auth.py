import boto3

# Initialize the Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')

# Create a collection to store face data (if not already created)
collection_id = "my_faces_collection"


def search_face(image_path, collection_id):
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    
    # Search for the face in the collection
    response = rekognition.search_faces_by_image(
        CollectionId=collection_id,
        Image={'Bytes': image_bytes},
        MaxFaces=1,
        FaceMatchThreshold=85  # Confidence threshold
    )

    # Display search results
    for match in response['FaceMatches']:
        print(match)
        print(f"Match found: FaceId={match['Face']['FaceId']}, Confidence={match['Face']['Confidence']}")

# Search for a face
search_face("/Users/hash/work/satschel/search_by_face_poc/images/auth.png", collection_id)