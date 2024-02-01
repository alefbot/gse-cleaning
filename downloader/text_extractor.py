import argparse
import asyncio
import base64
import json
import logging
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import quote

import aiohttp
import langdetect
import zstandard as zstd
from newspaper import Article
from zstandard import ZstdError

MAX_RETRIES = 2
NUM_PROCESSES = 20
FILE_PATH_204 = 'gse-text-data/urls_with_204'
MAIN_FILE_PATH = 'gse-text-data/main_files'
ARCHIVER_ADDRESS = os.getenv("ARCHIVER_ADDRESS")
LOG_FILENAME = "file.log"
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
    handlers=[logging.FileHandler(LOG_FILENAME), logging.StreamHandler()],
)

def modify_url(url):
    if url[:8] == "https://":
        url_part = url[8:]
        quoted_url = quote(url_part)
        url_modified = "https://" + quoted_url
    elif url[:7] == "http://":
        url_part = url[7:]
        quoted_url = quote(url_part)
        url_modified = "http://" + quoted_url
    else:
        url_modified = quote(url)
    return url_modified


def filter_url(url):
    if "aparat" in url:
        return ""
    else:
        return url    
    

def save_text(**kwargs):
    with open(kwargs['file_path'], "a+", encoding='utf-8') as f:
        f.write(json.dumps(kwargs, ensure_ascii=False))
        f.write('\n')
    

def text_language(txt):
    language_text = langdetect.detect(txt)
    return language_text


def extract_text(url, html):
    article: Article = Article(url, fetch_images=False)
    article.download_state = 2 
    article.download(input_html=html)
    article.parse()
    if len(article.text) < 1000:
        return ""
    text = re.sub(r"\n{2,}", "\n\n", article.text)
    return text


def decompress_zst_file(input_filename, output_filename=None):
    try:
        with open(input_filename, 'rb') as compressed_file:
            with zstd.open(compressed_file, 'rb') as decompressed_file:
                if output_filename:
                    with open(output_filename, 'wb') as output_file:
                        output_file.write(decompressed_file.read())
                else:
                    return decompressed_file.read().decode().split("\n")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {input_filename}") from e
    except zstd.ZstdError as e:
         raise ZstdError(f"Error decompressing file: {input_filename}") from e


async def send_urls(session, **kwargs):
    async def send_url(**kwargs):

        await asyncio.sleep(0.01) 
        data = {"url": kwargs['url']}
        headers = {'Content-type': 'application/json'}
        # async with async_timeout.timeout(5):
        try:
            async with session.post(ARCHIVER_ADDRESS, headers=headers, data=json.dumps(data)) as response:
                max_retries = 2  # Set the maximum number of retries
                retry_count = 0

                while response.status != 200 and retry_count < max_retries:
                    retry_count += 1
                    logging.info(f"Retrying request for {kwargs['url']} (attempt {retry_count}/{max_retries})")
                    # await asyncio.sleep(1)  # Optional delay between retries
                    data = {"url": modify_url(kwargs['url'])}
                    response = await session.post(ARCHIVER_ADDRESS, headers=headers, data=json.dumps(data))
                    
                if response.status == 200:
                    response_ = await response.read()
                    html = base64.b64decode(json.loads(response_)['result']['rawHtml'])
                    text = extract_text(kwargs['url'], html)
                    if text != "":                            
                        if text_language(text) == 'fa':
                            save_text(text=text, file_path=kwargs['file_path'], url=kwargs['url'])
                            logging.info(f"Response text for {kwargs['url']}")  
                        else:
                            logging.info(f"Response text for is not FARSI")
                    else:
                        logging.info(f"Response text for is None")
                elif response.status == 204:
                    logging.error(f"Error sending URL Status {response.status}")
                    save_text(url=kwargs['url'], file_path=kwargs['file_path_204'])
                else:
                    logging.error(f"Error sending URL Status {response.status}") 
        except asyncio.TimeoutError:
            logging.error(f"Timeout error for URL {kwargs['url']}") 
        except Exception as e:
            logging.error(f"Unexpected error for URL {kwargs['url']}: {traceback.format_exc()}") 
            
    executor = ProcessPoolExecutor(NUM_PROCESSES)
    tasks = [asyncio.create_task(send_url(url=url, file_path=kwargs['file_path'])).set_executor(executor) for url in kwargs['urls'] if filter_url(url) != ""]
    await asyncio.gather(*tasks)


async def receive_html(**kwargs):
    async with aiohttp.ClientSession() as session:
        logging.info("Starting URL sending process...")  
        await send_urls(session, **kwargs)
        logging.info("URL sending process completed.") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='gse text extractor',
                    description='This module can be used for downloading and extracting text from gse indices',
                    )
    parser.add_argument("--file_path", "-f", help="the .zstd file should be added to this folder", required=True)
    args = parser.parse_args()
    os.makedirs(MAIN_FILE_PATH, exist_ok=True)
    os.makedirs(FILE_PATH_204, exist_ok=True)
    save_file_path = f'{MAIN_FILE_PATH}/{re.sub("z8.zst", "jsonl", os.path.basename(args.file_path))}'
    file_path_204 = f'{FILE_PATH_204}/{re.sub("z8.zst", "jsonl", os.path.basename(args.file_path))}'
    urls = decompress_zst_file(args.file_path)
    if Path(save_file_path).is_file():
        seen_urls = [json.loads(_dict)['url'] for _dict in open(save_file_path, "r").readlines()]
        urls = set(urls) - set(seen_urls)
    if Path(file_path_204).is_file():
        urls_with_204_status_code = [json.loads(_dict)['url'] for _dict in open(file_path_204, "r").readlines()]
        urls = set(urls) - set(urls_with_204_status_code)
    logging.info(f"number of all URLs {len(urls)}")
    asyncio.run(receive_html(urls=urls, file_path=save_file_path, file_path_204=file_path_204))
