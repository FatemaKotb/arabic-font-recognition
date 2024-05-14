import requests

# Open the image file in binary mode
# This is necessary because the file needs to be sent over HTTP, and HTTP can only send text
with open('../data/raw/fonts-dataset/IBM Plex Sans Arabic/5.jpeg', "rb") as file:
    # Send a POST request to the server
    # The file is included in the 'files' parameter, which allows you to send files in a POST request
    response = requests.post("http://localhost:8000/predict/", files={"file": file})

# Print the response
print(response.json())