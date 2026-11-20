"""
Search input queries and downloads high resolution images.
Developed by <amin.aria@gmail.com>
"""

#from linkup import LinkupClient

from customSearchMethods import fetch_bing_highres_images, fetch_google_highres
CSE_URL = "https://cse.google.com/cse?cx=b4affdd81a66f4d8f"


import os, time
import requests
import hashlib
from PIL import Image
from io import BytesIO

#AA#> Concept Extraction
import openai
from utils import *
import pandas as pd

#AA#> DuckDuckGo
from ddgs import DDGS

from imageUsefulness import *

#AA#> Input Params
promptsPath = os.environ.get("RAGARTIST_PROMPTS_XLSX", "captions.xlsx")
conCol = 'Persian_Concepts'
download_path = 'fromWeb'

# OpenAI Connection
openai.api_key = os.environ.get("OPENAI_API_KEY")
gpt_client = openai.OpenAI()

def fetch_duckduckgo_images_no_browser(query, max_results=10):

    headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/122.0 Safari/537.36"
            }

    results = DDGS().images(query, max_results=max_results)

    return [r["image"] for r in results]

def persist_image(folder_path, url):
    try:
        image_content = requests.get(url, timeout=10).content
        image = Image.open(BytesIO(image_content)).convert('RGB')
        os.makedirs(folder_path,exist_ok=True)
        file_path = os.path.join(folder_path, f"{hash(url)}.jpg")
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG")
        print(f"✅ Saved: {file_path}")
    except Exception as e:
        print(f"❌ Failed to save {url}: {e}")

persianPrompts = pd.read_excel(promptsPath,index_col='id')
if conCol not in persianPrompts.columns: persianPrompts[conCol] = ''

for id in persianPrompts.index[656:701]:
    prompt = persianPrompts.loc[id,'Persian']
    english = persianPrompts.loc[id,'English']
    print('\n\n Searching concepts for: \n',prompt)

    #AA#> Extract Concepts from image caption (query)
    imgConcepts = (
        f'What visual concepts does a generative model need to know to generate an image described by the prompt "{prompt}"?\n'
        'The concepts should be things like objects that should appear in the image, the style of it, etc.'
        'Consider original language of the prompt as the cultural and geographical context of extracted concepts'
        'For example, if the prompt is "بشقاب پرنده ها در حال پرواز بر روی پل تاریخی در اصفهان", 3 concepts would be: بشقاب پرنده ها, نمای پل تاریخی اصفهان, and ژانر علمی-تخیلی.'
        'Do NOT include numbers or left or right in the concepts.'
        'In your answer only provide the concepts, each in a separate line. '
        'Return concepts in the original language of the prompt. Seperate concepts by \n')

    concepts = message_gpt_w_error_handle(imgConcepts, gpt_client, image_paths = [], context_msgs = [], max_tries=3)
    concepts = concepts.replace('\u200c',' ').split('\n')
    persianPrompts.loc[id,conCol] = str(concepts)

    if not download_path:
        persianPrompts.to_excel(promptsPath.split('.')[0]+'_out.xlsx')
        continue

    for j,concept in enumerate(concepts):
        print(f'Searching {concept}:...')
        #urls = fetch_duckduckgo_images_no_browser(concept, max_results = 10)
        urls = fetch_bing_highres_images(concept,5) #fetch_google_highres(concept,5)
        print(urls)

        #AA#> Apparently, servers do not let chatGPT download images!
        #top_images = screen_image_urls(urls, concept, prompt, top_k=2)
        #print('Top Images: ', json.dumps(top_images, indent=2, ensure_ascii=False))

        # Download and save
        for i,url in enumerate(urls):
            try:
                group = int(id[-3:])//50
                imgPath = download_path + f'/{group}/{english}_{j}'
                persist_image(imgPath,url)
                print(f'{url} is downloaded at {imgPath}')
            except Exception as e:
                print(f'Failed to download {url}: {e}')
            time.sleep(1)

        time.sleep(10)

    persianPrompts.to_excel(promptsPath.split('.')[0]+'_out.xlsx')
    print(f'Output excel is updated...')
