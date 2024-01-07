import re


class Patterns:
    PERSIAN_SIGN_IN = r"\s*وارد شوید[\s\.]*"
    YOUTUBE_TAG = r"\s*\{youtube+.*\}"
    PERSIAN_READ_MORE = r"\s*بیشتر بخوانید[\.\s\:]*"
    PERSIAN_CLICK = r"\s*(برای|می توانید برای|شما می توانید برای)+.*(نصب|کلیک|ثبت نام) کنید+(\s|\.)*"
    PERSIAN_REFERENCE = r"منبع[\s\:]*"
    CURRENCY_SYMBOLS = r"[$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6]"
    UNWANTED_ASCII = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]"
    PERSIAN_QUOTE = r"^[\s]*(به نقل از|به گزارش)[\w\s]+\،+$"
    URL_REGEX = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'
    USERNAME = r'(^|\w{1,15})@(\w{1,15})\b|(\w{1,15})@'
    IRAN_PHONE_NUMBER = r'((\+|00)98|0)\d{10}'
    CITATION = r'\[\d\]'
    START_WITH_NUMBER = r'[^\d\s]+'
    word_counters = r"(^|\s)\d+\.?\s"
