import uvicorn
from fastapi import FastAPI, Form
import logging
import os

from filter_text import *

PORT = 5001
APP_VERSION = "0.0.0"
APP_NAME = "filtering-service"


logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)


def apply_logger_handlers() -> None:
    """Apply generic configuration of logger."""
    # create handlers
    c_handler = logging.StreamHandler()

    log_filename = f"./logs/{APP_NAME}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    f_handler = logging.FileHandler(log_filename)

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # create formatters and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - " "%(levelname)s - %(message)s")

    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


apply_logger_handlers()

app = FastAPI(    
    title=APP_NAME,
    version=APP_VERSION,
    contact={
        "name": "Mohammad Javad Taheri (_hajagha)",
        "email": "mj.taheri1996@gmail.com",
    },
)

def filter_and_process_valid_examples(examples, **kwargs):

    filtered_texts = [
        reduce(lambda x, y: y(x), kwargs['document_wise_filtering'], example)
        for example in examples
        if all(filter_func(example) for filter_func in kwargs['repetition_removal'])
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
                    normalized_line = reduce(lambda x, y: y(x), kwargs['document_wise_filtering'], line)
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
    


@app.post('/')
async def main(html: str = Form(...)):
    
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
    html = extract_text(html)
    if html == "":
        return [""]
    text_dict = filter_and_process_valid_examples([html],
                                                  repetition_removal=repetition_removal,
                                                  document_wise_filtering=document_wise_filtering,
                                                  line_wise_filtering=line_wise_filtering
                                                  )
    return text_dict['text']


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=PORT)

