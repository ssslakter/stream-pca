import os
import urllib.request

def download_test_data():
    videos = {
        "hall.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "traffic.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "station.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4"
    }
    
    for name, url in videos.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, name)
            print("Done.")
        else:
            print(f"{name} already exists.")

if __name__ == "__main__":
    download_test_data()