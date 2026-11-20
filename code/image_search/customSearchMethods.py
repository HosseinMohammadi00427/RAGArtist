import os,json
import time, random
import requests
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


# ========== CONFIG ==========
DRIVER_PATH = "./chromedriver"  # update this!
CSE_URL = "https://cse.google.com/cse?cx=b4affdd81a66f4d8f"  # your public CSE URL
SAVE_DIR = "./cse_images"
MAX_IMAGES = 10
HEADLESS = True

# ============================

def setup_driver(headless=True):
    """Set up Chrome with optional headless mode."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    service = Service(DRIVER_PATH)
    return webdriver.Chrome(service=service, options=options)

def fetch_bing_highres_images(query: str, max_images: int, driver_path = DRIVER_PATH, headless=True):
    # --- Setup Chrome ---
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(driver_path)
    wd = webdriver.Chrome(service=service, options=options)

    # --- Go to Bing Images ---
    scroll_pause = 2
    search_url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&form=HDRSC2"
    wd.get(search_url)
    time.sleep(scroll_pause)

    image_urls = set()
    results_start = 0

    while len(image_urls) < max_images:
        # Find all image containers
        image_elements = wd.find_elements(By.CSS_SELECTOR, "a.iusc")

        for image_element in image_elements[results_start:]:
            try:
                # Each tile has metadata in the 'm' attribute
                meta_json = image_element.get_attribute("m")
                meta = json.loads(meta_json)
                image_url = meta.get("murl")  # <- the high-res image URL
                if image_url and image_url.startswith("http"):
                    image_urls.add(image_url)
                    print(f"🖼️ Found: {image_url}")
                    if len(image_urls) >= max_images:
                        break
            except Exception as e:
                continue

        # Scroll to load more
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

        results_start = len(image_elements)

        # Stop scrolling if we’re not getting new results
        if results_start >= max_images:
            break

    wd.quit()
    print(f"✅ Collected {len(image_urls)} high-res image URLs")
    return list(image_urls)


def fetch_google_highres(query: str, max_images: int, driver_path = DRIVER_PATH, headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    service = Service(driver_path)
    wd = webdriver.Chrome(service=service, options=options)

    search_url = f"https://www.google.com/search?tbm=isch&q={query.replace(' ', '+')}"
    wd.get(search_url)
    time.sleep(2)

    image_urls = set()
    thumbnails = wd.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")

    for thumb in thumbnails[:max_images * 2]:
        try:
            thumb.click()
            time.sleep(1)
            images = wd.find_elements(By.CSS_SELECTOR, "img.sFlh5c.FyHeAf")
            for image in images:
                src = image.get_attribute("src")
                if src and "http" in src and not src.startswith("data"):
                    image_urls.add(src)
                    if len(image_urls) >= max_images:
                        break
        except Exception:
            continue
        if len(image_urls) >= max_images:
            break

    wd.quit()
    print(f"✅ Collected {len(image_urls)} URLs")
    return list(image_urls)



def fetch_thumbnails_from_cse(search_term, cse_url, max_links=5, delay=1.5,headless=True):

    """Click thumbnails and extract high-res image URLs from Google CSE."""
    wd = setup_driver(headless)
    wd.get(f"{cse_url}&q={search_term}")
    time.sleep(3)

    image_urls = set()
    seen = set()

    # Scroll to load enough thumbnails
    for _ in range(3):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)

    thumbnails = wd.find_elements(By.CSS_SELECTOR, "img")

    for thumb in thumbnails:
        if len(image_urls) >= max_links:
            break

        try:
            src = thumb.get_attribute("src")
            if not src or src in seen:
                continue
            seen.add(src)

            # Click to open high-res version
            wd.execute_script("arguments[0].click();", thumb)
            time.sleep(random.uniform(1.5, 3.0))

            # Find large image inside preview container
            large_images = wd.find_elements(By.CSS_SELECTOR, "img[src^='http']")
            for img in large_images:
                full_src = img.get_attribute("src")
                if full_src and "encrypted-tbn0" not in full_src:  # skip thumbnails
                    image_urls.add(full_src)
                    print(f"✅ Found high-res: {full_src[:100]}")
                    break

        except Exception as e:
            print(f"⚠️ Error clicking thumbnail: {e}")
            continue

    wd.quit()
    print(f"🎉 Collected {len(image_urls)} high-resolution images.")
    return list(image_urls)
