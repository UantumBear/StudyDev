# utils/text_filters.py
import re

# 이모지/픽토그램 범위 (대표적 블록들)
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF"  # Misc Symbols and Pictographs ~ Symbols and Pictographs Extended-A
    r"\U00002600-\U000026FF"   # Misc Symbols
    r"\U00002700-\U000027BF"   # Dingbats
    r"]+",
    flags=re.UNICODE,
)

def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)