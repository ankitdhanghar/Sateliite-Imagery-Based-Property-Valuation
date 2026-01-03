import requests as rq
from PIL import Image as Img
from io import BytesIO as Bio
import pandas as pd
import os as fs
import time as tm


class MapboxImageCollector:
    """Handles retrieval and validation of satellite images from Mapbox."""

    def __init__(self, token_key, dimension='256x256', zoom_value=18):
        """
        Initialize the image collector.
        """
        self.token_key = token_key
        self.endpoint_url = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static'
        self.dimension = dimension
        self.zoom_value = zoom_value

        print("Mapbox Image Collector initialized")
        print(f"Settings -> Size: {dimension}, Zoom: {zoom_value}")


    def fetch_image(self, lat_value, lon_value, record_id, output_file):
        """
        Download a single satellite image.

        Returns:
            bool
        """
        try:
            request_url = (
                f"{self.endpoint_url}/"
                f"{lon_value},{lat_value},{self.zoom_value}/"
                f"{self.dimension}?access_token={self.token_key}"
            )

            api_response = rq.get(request_url, timeout=10)

            if api_response.status_code == 200:
                image_obj = Img.open(Bio(api_response.content))
                image_obj.save(output_file, 'JPEG', quality=95)
                return True
            else:
                print(f"Download failed for ID {record_id}: HTTP {api_response.status_code}")
                return False

        except Exception as error_msg:
            print(f"Error for ID {record_id}: {str(error_msg)}")
            return False


    def build_image_index(self, source_frame, image_dir, csv_output):
        """
        Generate a CSV linking records to image paths.

        Returns:
            pd.DataFrame
        """
        print("Generating image index file...")

        index_frame = source_frame[['id', 'lat', 'long']].copy()

        index_frame['file_location'] = index_frame['id'].apply(
            lambda identifier: fs.path.join(image_dir, f'property_{identifier}.jpg')
        )

        index_frame['file_available'] = index_frame['file_location'].apply(fs.path.exists)

        index_frame.to_csv(csv_output, index=False)

        available_count = index_frame['file_available'].sum()
        print(f"Images present: {available_count}/{len(index_frame)}")
        print(f"Index saved at: {csv_output}")

        return index_frame
