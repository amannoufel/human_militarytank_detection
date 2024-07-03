import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

def fetch_image_urls(query, max_links_to_fetch):
    search_url_template = "https://www.google.com/search?hl=en&tbm=isch&q={q}&start={start}"
    image_urls = []
    start = 0

    while len(image_urls) < max_links_to_fetch:
        search_url = search_url_template.format(q=quote(query), start=start)
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        new_image_urls = [img.get("src") for img in soup.find_all("img") if img.get("src") and img.get("src").startswith("http")]
        if not new_image_urls:
            print("No more images found.")
            break
        
        image_urls.extend(new_image_urls)
        start += 20  # Google Images paginates in increments of 20

        if len(image_urls) >= max_links_to_fetch:
            image_urls = image_urls[:max_links_to_fetch]
            break

    return image_urls

def download_images(image_urls, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i, url in enumerate(image_urls):
        try:
            img_data = requests.get(url).content
            with open(os.path.join(save_path, f"tank_{i}.jpg"), 'wb') as img_file:
                img_file.write(img_data)
            print(f"Downloaded {i + 1} images")
        except Exception as e:
            print(f"Could not download image {i + 1}. Error: {e}")

query = "military tank"
max_links_to_fetch = 5000
save_path = "tanks_dataset"

image_urls = fetch_image_urls(query, max_links_to_fetch)
download_images(image_urls, save_path)
