# from hazm import Normalizer
import argparse
import concurrent
import concurrent.futures
import html
import logging
import logging.config
import os
import re
import unicodedata
from collections import Counter
from functools import reduce
from itertools import groupby
from string import punctuation

import datasets
import langdetect
from datasets import load_dataset
from newspaper import Article
from patterns import Patterns
from tokenizers import normalizers
from tokenizers.normalizers import NFC, NFD, NFKC, NFKD

#TODO: remove CSS and Javascript
# from collection.abs import sequence. didn't still contributed in parsivar normalizer

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
    }
)
logging.basicConfig(
    filename="temp",
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
MIN_LINE_WORDS = 10


Unicode_normalizer = normalizers.Sequence([NFD(), NFKD(), NFC(), NFKC()])        


def extract_text(html):
    article: Article = Article("temp.com", fetch_images=False)
    article.download_state = 2 
    article.download(input_html=html)
    article.parse()
    if len(article.text) < 1000 or langdetect.detect(article.text) != "fa":
        return ""
    text = re.sub(r"\n{2,}", "\n\n", article.text)
    return text


def fix_html(txt):
    "From fastai: 'Fix messy things we've seen in documents'"
    txt_normalized = txt.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', ' ').replace(' @.@ ', '.').replace(' @-@ ', '-').replace('...', ' …')
    htmp_unscape = html.unescape(txt_normalized)
    return htmp_unscape


# what is so category? symbol other means i think
def remove_unicode_symbols(txt):
    normalize_txt = ""
    for ch in txt:
        if unicodedata.category(ch)[0] != "So":
            normalize_txt += ch

    return normalize_txt


def standardise_punc(txt):
    transl_table = dict([(ord(x), ord(y))
                         for x, y in zip(u"‘’´“”–",  u"'''\"\"-")])
    txt_normalized = txt.translate(transl_table)
    return txt_normalized


def remove_duplicate_punctuation(txt):
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


# from mj: as naser said, I think should even consider
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


def remove_quote(txt):
    normalized_txt = re.sub(Patterns.PERSIAN_QUOTE, "", txt)
    return normalized_txt


def youtube_tags(txt):
    normalized_txt = re.sub(Patterns.YOUTUBE_TAG, " ", txt)
    return normalized_txt 


def remove_citation(txt):
    normalized_text = re.sub(Patterns.CITATION, "", txt)
    return normalized_text


# here we should use filter rather than map
def remove_documents_by_word_length(txt):
    lines = txt.split("\n")
    words = sum([line.split() for line in lines], [])  # Split text into words

    word_lengths = [len(word) for word in words]  # Get lengths of all words
    mean_word_length = sum(word_lengths) / len(word_lengths)  # Calculate mean word length

    if not 2 <= mean_word_length <= 9:  # Check if mean word length is outside the range
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
    txt = re.sub("\n\n", "\n", txt)
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


def calculate_line_duplicate_char_fractions(text, threshold=0.2):
    # Calculate duplicate fractions for lines
    text = re.sub("\n\n", "\n", text)
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(lines) > 0:
        line_duplicate_char_fraction = 1 - sum(len(line) for line in unique_lines) / sum(len(line) for line in lines)
    else:
        line_duplicate_char_fraction = 0
    
    if line_duplicate_char_fraction > threshold:
        return False
    return True


def calculate_paragraph_duplicate_char_fractions(text, threshold=0.2):
    # Calculate duplicate fractions for paragraphs
    paragraphs = text.split('\n\n')
    unique_paragraphs = set(paragraphs)
    if len(paragraphs) > 0:
        paragraph_duplicate_char_fraction = 1 - sum(len(paragraph) for paragraph in unique_paragraphs) / sum(len(paragraph) for paragraph in paragraphs)
    else:
        paragraph_duplicate_char_fraction = 0    

    if paragraph_duplicate_char_fraction > threshold:
        return False
    return True


def generate_ngrams(text, n):
    normalized_text = re.sub(r"\n+", " ", text.strip())
    ngrams = [normalized_text[i:i+n] for i in range(len(normalized_text)-n+1)]
    return ngrams


def calculate_top_ngram_char_fraction(text, threshold=0.2):
    for n in range(2, 5):
        ngrams = generate_ngrams(text, n)
        ngram_counts = Counter(ngrams)
        most_common_ngram, most_common_count = ngram_counts.most_common(1)[0]
        fraction = most_common_count * len(most_common_ngram) / len(text)
        if fraction > threshold:
            return False
        threshold -= 0.02
    return True


def calculate_duplicated_ngram_char_fraction(text, threshold=0.15):
    for n in range(5, 11):
        ngrams = generate_ngrams(text, n)
        ngram_counts = Counter(ngrams)
        duplicate_n_gram_chars = set(n_gram for n_gram, count in ngram_counts.items() if count > 1)
        unique_duplicate_n_gram_chars = set(char for n_gram in duplicate_n_gram_chars for char in n_gram)
        fraction = len(unique_duplicate_n_gram_chars) / len(text)
        if fraction > threshold:
            return False
        threshold -= 0.01
    return True

    
def remove_reference(txt):
    normalized_text = re.sub(Patterns.PERSIAN_REFERENCE, "", txt)
    return normalized_text
            
            
def remove_read_more(txt):
    normalized_text = re.sub(Patterns.PERSIAN_READ_MORE, "", txt)
    return normalized_text


def remove_sign_in(txt):
    normalized_text = re.sub(Patterns.PERSIAN_SIGN_IN, "", txt)
    return normalized_text


def remove_click(txt):
    normalized_text = re.sub(Patterns.PERSIAN_CLICK, "", txt)
    return normalized_text

            
def filter_and_process_valid_examples(examples):
    
    filtered_texts = [
        reduce(lambda x, y: y(x), document_wise_filtering, example)
        for example in examples
        if all(filter_func(example) for filter_func in repetition_removal)
    ]
    
    tmp_lst = []
    for example in filtered_texts:
        main_lines = []
        for line in example.strip().split('\n'):
            if line.strip() != "":
                words = line.split()
                words_length = len(words)
                if words_length <= 1 and words[0] != "":
                    main_lines.append("")
                elif words_length < MIN_LINE_WORDS:
                    normalized_line = reduce(lambda x, y: y(x), document_wise_filtering, line)
                    if len(normalized_line) <= 1:
                        main_lines.append("")
                    # adding extra \n is a good idea i think. here we have consecutive \n\n 
                    else:
                        main_lines.append(line)
                else:
                    main_lines.append(line)
            else:
                main_lines.append("")
            
        tmp_lst.append("\n".join(main_lines))

    return {"text": tmp_lst}
    

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
    repetition_removal = [
        remove_documents_by_word_length,
        remove_documents_by_symbol_ratio,
        remove_documents_bullet_point_ellipsis,
        filter_documents_by_alphabetic_words,
        filter_documents_by_line_repetition,
        filter_documents_by_paragraph_repetition,
        calculate_line_duplicate_char_fractions,
        calculate_paragraph_duplicate_char_fractions,
        calculate_top_ngram_char_fraction,
        calculate_duplicated_ngram_char_fraction
    ]
    
    document_wise_filtering = [
        fix_html,
        remove_unicode_symbols,
        standardise_punc,
        remove_duplicate_punctuation,
        remove_unicode,
        remove_currency_symbols,
        remove_wierd_unicode,
        remove_unwanted_ascii,
        youtube_tags,
        remove_citation,
    ]
    
    line_wise_filtering = [
        remove_click,
        remove_reference,
        remove_read_more,
        remove_sign_in,
    ]
    
    os.makedirs(args.save_path, exist_ok=True)
    dataset = load_dataset(
        args.dataset_path,
        num_proc=20,
        cache_dir="/mnt/data"
        )
    
    print("dataset was loaded")
    cleaned_data = dataset.map(filter_and_process_valid_examples,
                               #num_proc=20,
                               num_proc=os.cpu_count() if args.max_workers is None else args.max_workers,
                               batched=True,
                               writer_batch_size=100000,
                               keep_in_memory=True,
                               input_columns=args.column_name,
                               remove_columns=dataset.column_names
                               )
    
    cleaned_data.save_to_disk(args.save_path)
    print(cleaned_data)
    
