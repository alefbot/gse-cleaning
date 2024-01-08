# from hazm import Normalizer
import argparse
import concurrent
import concurrent.futures
import html
import json
import logging
import logging.config
import os
import re
import traceback
import unicodedata
from glob import glob
import multiprocessing as mp
import time
import collections
from itertools import groupby
from string import punctuation
from functools import partial
from threading import Lock
from unidecode import unidecode
# from zarebin_normalizer.normalizer import Normalizer

import datasets
from datasets import load_dataset
from fastcore.basics import listify
from fastcore.utils import compose
from tokenizers import normalizers
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD
from tqdm import tqdm

from normalize import Normalizer
from config import config
from patterns import Patterns
 
#TODO: remove CSS and Javascript
# from collection.abs import sequence. didn't still contributed in parsivar normalizer

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
    }
)
logging.basicConfig(
    filename=config["logging"]["name"],
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

datasets.logging.set_verbosity_info()
datasets.logging.DEBUG

INPUT_DIRECTORY = "/mnt/data/llm-data/src/preprocess/data/lang_detect/processed/"
OUTPUT_DIRECTORY = "/mnt/data/llm-data/src/preprocess/data/string_cleaning_temp/"
CHAR_NUMBER_RATIO = .6
MIN_DOCUMEMTN_LENGTH = 25


normalizer = Normalizer(remove_punctuation=False, word_number_separation=False)
# remove different kind of uincode
# control_char_regex = re.compile(r'[\r\n\t]+')

Unicode_normalizer = normalizers.Sequence([NFD(), NFKD(), NFC(), NFKC()])        

def fix_html(txt):
    "From fastai: 'Fix messy things we've seen in documents'"
    txt_normalized = txt.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', ' ').replace(' @.@ ', '.').replace(' @-@ ', '-').replace('...', ' …')
    htmp_unscape = html.unescape(txt_normalized)
    return htmp_unscape


def arabic_to_english_numbers(txt):
    txt_normalized = re.sub(r'\d', lambda x: unidecode(x.group()), txt)
    return txt_normalized


def add_space_between_numbers(txt):
    txt_normalized = re.sub(r'\d{2,}', lambda x: " ".join(x.group()), txt)
    return txt_normalized


def half_to_full_space(txt):
    txt_normalized = re.sub('\u200c', " ", txt)
    return txt_normalized


# what is so category? symbol other means i think
def remove_unicode_symbols(txt):
    normalize_txt = ""
    for ch in txt:
        if unicodedata.category(ch)[0] != "So":
            normalize_txt += ch

    return normalize_txt


def standardise_punc(txt):
    transl_table = dict([(ord(x), ord(y))
                         for x, y in zip(u"‘’´“”–-",  u"'''\"\"--")])
    txt_normalized = txt.translate(transl_table)
    # e = re.sub(r"[^a-zA-Z0-9ÖÄÅöäå .,'%&€$=*@+;<>/()!?%:-]", " ", e)

    return txt_normalized


def remove_news_tags(txt):
    normalized_txt = re.sub(r"(<[A-Z].+?>)|(</[A-Z].+?>)", "", txt)
    return normalized_txt


# urls not important? because are english?
def replace_urls(txt):
    # remove urls?
    normalized_txt = re.sub(Patterns.URL_REGEX, "", txt)
    return normalized_txt


def replace_usernames(txt):
    occ = txt.count('@')
    for _ in range(occ):
        txt = txt.replace('@<user>', "")
        # replace other user handles by filler
        txt = re.sub(Patterns.USERNAME, "", txt)
        # add spaces between, and remove double spaces again
        # e = e.replace(filler, f' {filler} ')
        txt = ' '.join(txt.split())

    return txt


def remove_duplicate_words_punctuation(txt):
    # from mj: what is it doing?
    txt = re.sub(r'\b(\w+)( \1\b)+', r'\1', txt)
    punc = set(punctuation+"؟!،")
    newtext = []
    for k, g in groupby(txt):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)

    normalized_text = ''.join(newtext)
    return normalized_text


def remove_unicode(txt):
    txt = Unicode_normalizer.normalize_str(txt)
    normalized_txt = html.unescape(txt)
    return normalized_txt


def replace_phone_numbers(txt):
    # e = re.sub(r"", "", e)
    normalized_txt = re.sub(Patterns.IRAN_PHONE_NUMBER, "", txt)
    return normalized_txt


# from mj: as naser, I think should even consider
def remove_currency_symbols(txt):
    normalized_txt = re.sub(Patterns.CURRENCY_SYMBOLS, "", txt)
    return normalized_txt


# from mj: don't know what are wired
def remove_wierd_unicode(txt):
    normalized_txt = re.sub(u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F680-\U0001F6FF\U0001F680-\U0001F6FF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f927\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\u3030\ufe0f\u2069\u2066\u2068\u2067]", "", txt)
    return normalized_txt


def remove_unwanted_ascii(txt):
    normalized_txt = re.sub(Patterns.UNWANTED_ASCII, "", txt)
    return normalized_txt


# from mj: really remove them? even for llm? # can be added 
def replace_char(txt):
    normalized_txt = re.sub(r"\||_|-", " ", txt)
    return normalized_txt 


def normalize_parsivar(txt):
    normalized_txt = normalizer.normalize_text(txt)
    return normalized_txt


def remove_quote(txt):
    normalized_txt = re.sub(Patterns.PERSIAN_QUOTE, "", txt)
    return normalized_txt


def youtube_tags(txt):
    normalized_txt = re.sub(Patterns.YOUTUBE_TAG, " ", txt)
    return normalized_txt 


def remove_citation(txt):
    normalized_text = re.sub(Patterns.CITATION, "", txt)
    return normalized_text


def start_with_number(txt):
    normalized_txt = re.sub(Patterns.START_WITH_NUMBER, "", txt)
    return normalized_txt

# here we should use filter rather than map
def remove_documents_by_word_length(txt):
    lines = txt.split("\n")
    words = sum([line.split() for line in lines], [])  # Split text into words

    word_lengths = [len(word) for word in words]  # Get lengths of all words
    mean_word_length = sum(word_lengths) / len(word_lengths)  # Calculate mean word length

    if not 3 <= mean_word_length <= 10:  # Check if mean word length is outside the range
        return False
    return True


def remove_documents_by_symbol_ratio(txt):
    words = txt.split()

    hash_count = txt.count("#")
    ellipsis_count = txt.count("...")

    if not words:  # Handle empty documents
        return False

    hash_ratio = hash_count / len(words)
    ellipsis_ratio = ellipsis_count / len(words)

    if hash_ratio > 0.1 or ellipsis_ratio > 0.1:
        return False
    return True

# here we should use filter rather than map
def remove_documents_bullet_point_ellipsis(txt):
    lines = txt.split("\n")
    bullet_count = 0
    ellipsis_count = 0
    total_lines = len(lines)

    for line in lines:
        if line.strip().startswith("- ") or line.strip().startswith("\u2022"):  # Check for bullet point
            bullet_count += 1
        if line.strip().endswith("..."):  # Check for ellipsis
            ellipsis_count += 1

    if (bullet_count / total_lines) > 0.9 or (ellipsis_count / total_lines) > 0.3:
        return False
    return True


def filter_documents_by_alphabetic_words(txt):
    words = txt.split()

    alphabetic_word_count = 0
    total_word_count = len(words)

    for word in words:
        if any(char.isalpha() for char in word):  # Check if word contains at least one alphabetic character
            alphabetic_word_count += 1

    alphabetic_word_ratio = alphabetic_word_count / total_word_count

    if alphabetic_word_ratio < 0.8:
        return False
    return True


def filter_documents_by_line_repetition(txt, threshold=0.3):
    
    txt_normalized = re.sub("\n\n", "\n", txt)
    lines = txt_normalized.split("\n")

    unique_lines = set(lines)  # Remove duplicates
    duplicate_count = len(lines) - len(unique_lines)
    duplicate_fraction = duplicate_count / len(lines)

    if duplicate_fraction > threshold:
        return False
    return True


def filter_documents_by_paragraph_repetition(txt, threshold=0.3):

    paragraphs = txt.split("\n\n")  # Split text into paragraphs

    unique_paragraphs = set(paragraphs)
    duplicate_count = len(paragraphs) - len(unique_paragraphs)
    duplicate_fraction = duplicate_count / len(paragraphs)

    if duplicate_fraction > threshold:
        return False
    return True


def filter_line_character_fraction(txt, threshold=.2):
    normalized_txt = re.sub("\n\n", "\n", txt)
    lines = normalized_txt.split("\n")
    for line in lines:
        

def remove_reference(txt, mod="delete"):
    reference_match = re.search(Patterns.PERSIAN_REFERENCE, txt)
    if reference_match:
        if 0 == reference_match.start():
            if mod == "delete":  
                return ""
            elif mod == "replace":
                normalized_text = re.sub(Patterns.PERSIAN_REFERENCE, "", txt)
                return normalized_text
    return txt
            
            
def remove_read_more(txt, mod="delete"):
    persian_read_more_match = re.search(Patterns.PERSIAN_READ_MORE, txt)
    if persian_read_more_match:
        if persian_read_more_match.end() == len(txt.strip()):
            if mod == "delete":
                return ""
            elif mod == "replace":
                normalized_text = re.sub(Patterns.PERSIAN_READ_MORE, "", txt)
                return normalized_text
    return txt


def remove_sign_in(txt, mod="delete"):
    persian_sign_in = re.search(Patterns.PERSIAN_SIGN_IN, txt)
    if persian_sign_in:
        if persian_sign_in.end() == len(txt.strip()):
            if mod == "delete":
                return ""
            elif mod == "replace":
                normalized_text = re.sub(Patterns.PERSIAN_SIGN_IN, "", txt)
                return normalized_text
    return txt


def remove_click(txt, mod="delete"):
    persian_sign_in = re.search(Patterns.PERSIAN_CLICK, txt)
    if persian_sign_in:
        if persian_sign_in.end() == len(txt.strip()):
            if mod == "delete":
                return ""
            elif mod == "replace":
                normalized_text = re.sub(Patterns.PERSIAN_CLICK, "", txt)
                return normalized_text
    return txt

        
def process(examples):
    tmp_lst = []
    for example in listify(examples['text']):
        main_lines = []
        normalized_document = document_wise_filtering(example)
        for line in normalized_document.split('\n'):
            normalized_line = remove_click(line)
            if len(line.split()) < MIN_DOCUMEMTN_LENGTH:
                normalized_line = line_wise_filtering(normalized_line)
                # adding extra \n is a good idea i think. here we have consecutive \n\n 
                main_lines.append(normalized_line)
            else:
                main_lines.append(normalized_line)
            
        tmp_lst.append("\n".join(main_lines))
    
    examples['text'] = tmp_lst
    return examples
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Dataset Preprocessing") 
    parser.add_argument("--dataset_path",
                        type=str,
                        default="/mnt/data/llm-data/src/preprocess/data/lang_detect/processed",
                        help="Path of the dataset which contains text or gzip files"
                        )

    parser.add_argument("--save_path",
                        type=str,
                        default=OUTPUT_DIRECTORY,
                        help="Path to save the proccessed dataset"
                        )
    
    parser.add_argument("--max_workers",
                        type=int,
                        default=None,
                        help="max multi-process workers"
                        )

    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        help="where to cache the data"
                        )
    
    parser.add_argument("--column_name",
                        type=str,
                        default=None,
                        help="which column should be processed"
                        )
    args = parser.parse_args()
    document_wise_filtering = compose(
        fix_html,
        arabic_to_english_numbers,
        half_to_full_space,
        remove_unicode_symbols,
        standardise_punc,
        remove_news_tags,
        replace_urls,
        replace_usernames,
        remove_duplicate_words_punctuation,
        remove_unicode,
        replace_phone_numbers,
        remove_currency_symbols,
        remove_wierd_unicode,
        remove_unwanted_ascii,
        replace_char,
        normalize_parsivar,
        remove_quote,
        youtube_tags,
        remove_citation,
        start_with_number
    )
    
    line_wise_filtering = compose(
        remove_reference,
        remove_read_more,
        remove_sign_in,
    )
    
    os.makedirs(args.save_path, exist_ok=True)
    # data_files = os.path.join(args.dataset_path, "*.json")
    # path_json_files = glob(os.path.join(dataset_path, '**/*.json'), recursive=True)
    # data_files = [os.path.join(dataset_path, "CC-MAIN-20230610233020-20230611023020-00771/*.json"), os.path.join(dataset_path, "CC-MAIN-20230610233020-20230611023020-00781/*.json")]
    dataset = load_dataset(
        args.dataset_path,
        num_proc=20,
        cache_dir="/mnt/data"
        )
    
    print("dataset was loaded")
    cleaned_data = dataset['train'].map(process,
                               #num_proc=20,
                               num_proc=os.cpu_count() if args.max_workers is None else args.max_workers,
                               batched=True,
                               writer_batch_size=100000,
                               keep_in_memory=True,
                               input_columns=args.column_name
                               )
    
    cleaned_data.save_to_disk(args.save_path)
    print(cleaned_data)
    
