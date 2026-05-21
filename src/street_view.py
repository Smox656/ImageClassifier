import requests
import os


def download_image_street_view(api_key, lat, lng, filename):
    url = "https://maps.googleapis.com/maps/api/streetview"

    parameters = {
        "size": "400x400",
        "location": f"{lat},{lng}",
        "fov" : "90",
        "pitch": "0",
        "key": api_key
    }

    print(f"Downloading of the image for the coordinates {lat}, {lng}...")

    response = requests.get(url, params=parameters)

    if response.status_code == 200:
        with open(filename, 'wb') as file :
            file.write(response.content)
        print(f"Success : Image saved under {filename}")
    else:
        print(f"Error {response.status_code} : Impossible to download the image.")
        print(response.text)



if __name__ == "__main__":

    MA_CLE_API = "AIzaSy_TA_CLE_SECRETE_ICI" 
    

    latitude = 48.8584
    longitude = 2.2945
    
    if not os.path.exists("data/nouvelles_images"):
        os.makedirs("data/nouvelles_images")
        
    chemin_sauvegarde = "data/nouvelles_images/test_maison.jpg"
    
    telecharger_image_street_view(MA_CLE_API, latitude, longitude, chemin_sauvegarde)