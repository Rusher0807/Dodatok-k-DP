import os
import requests

# GitHub API URL for the 'benign' directory
api_url = "https://api.github.com/repos/iosifache/DikeDataset/contents/files/malware"

# Headers for the API request
headers = {
    "Accept": "application/vnd.github.v3+json"
}

# Make the API request
response = requests.get(api_url, headers=headers)
if response.status_code != 200:
    raise Exception(f"Failed to fetch directory contents: {response.status_code}")

# Parse the JSON response
files = response.json()

# Create the 'benign' directory if it doesn't exist
os.makedirs("malicious", exist_ok=True)

# Download each '.exe' file
for file in files:
    file_name = file["name"]
    download_url = file["download_url"]
    if file_name.endswith(".exe"):
        print(f"Downloading {file_name}...")
        file_response = requests.get(download_url)
        if file_response.status_code == 200:
            with open(os.path.join("malicious", file_name), "wb") as f:
                f.write(file_response.content)
        else:
            print(f"Failed to download {file_name}: {file_response.status_code}")
