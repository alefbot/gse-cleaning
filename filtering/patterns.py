import re


class Patterns:
    PERSIAN_SIGN_IN = r"^\s*وارد شوید[\s\.]*"
    YOUTUBE_TAG = r"\s*\{youtube+.*\}"
    PERSIAN_READ_MORE = r"\s*بیشتر (بخوانید|بدانید)...$"
    PERSIAN_CLICK = r"^(برای|می توانید برای|شما می توانید برای)+[\w\s]{3,15}(نصب|ثبت نام) کلیک کنید+(\s|\.)$"
    PERSIAN_REFERENCE = r"^منبع[\s\:]*"
    CURRENCY_SYMBOLS = r"[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]"
    UNWANTED_ASCII = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]"
    URL_REGEX = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    IRAN_PHONE_NUMBER = r'((\+|00)98|0)\d{10}'
    CITATION = r'\[\d+\]'
    COUNTERS = r"(^|\s)\d+\.?\s"
    NON_SYMBOL = r"[^\w\s\d]"
