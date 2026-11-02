import re
import string

import jieba  # for Chinese word segmentation


def preprocess_text(text: str) -> str:
    """
    Preprocess text: convert to lowercase, remove special characters, handle whitespace, etc.
    """
    # Convert to lowercase (only effective for English)
    text = text.lower()

    # Remove all punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Handle newlines and multiple spaces
    return re.sub(r'\s+', ' ', text).strip()


def is_chinese(text: str) -> bool:
    """Determine if the text is primarily Chinese"""
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) > len(text) * 0.3  # Consider as Chinese text if Chinese characters exceed 30%


def ngram_similarity(text1: str, text2: str, n: int = 3) -> float:
    """
    Calculate similarity between two texts using n-gram.
    For Chinese, perform word segmentation first.
    """
    # Determine if it's Chinese text
    chinese_text = is_chinese(text1) and is_chinese(text2)

    # Generate n-gram set
    def get_ngrams(text: str, n: int, chinese: bool) -> set:
        if chinese:
            # Chinese word segmentation
            words = list(jieba.cut(text))
            ngrams = set()
            for i in range(len(words) - n + 1):
                ngrams.add(' '.join(words[i : i + n]))
            return ngrams
        # Process English by characters
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i : i + n])
        return ngrams

    ngrams1 = get_ngrams(text1, n, chinese_text)
    ngrams2 = get_ngrams(text2, n, chinese_text)

    # Calculate intersection and union
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    # Avoid division by zero
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def contains_question(question: str, doc: str, threshold: float = 0.6) -> tuple[bool, float]:
    """
    Determine if the document contains the question (considering fuzzy matching).
    Returns: (whether contained, similarity score)
    """
    # Preprocess
    processed_question = preprocess_text(question)
    processed_doc = preprocess_text(doc)

    # Adjust n value based on text type
    chinese_text = is_chinese(processed_question)
    n = 2 if chinese_text else 3  # Use 2-gram for Chinese, 3-gram for English

    # Calculate overall similarity
    overall_similarity = ngram_similarity(processed_question, processed_doc, n)

    # If overall similarity is high enough, return directly
    if overall_similarity >= threshold:
        return True, overall_similarity

    # Otherwise, try to find segments in the document similar to the question
    # For longer documents, we can use a sliding window to find matches
    question_length = len(processed_question)
    doc_length = len(processed_doc)

    # If the document is shorter than the question, return directly
    if doc_length < question_length:
        return False, overall_similarity

    # Sliding window check
    max_similarity = 0.0
    window_size = question_length
    step = max(1, window_size // 10)  # Step size is 1/10 of the window size

    for i in range(0, doc_length - window_size + 1, step):
        window = processed_doc[i : i + window_size]
        similarity = ngram_similarity(processed_question, window, n)
        if similarity > max_similarity:
            max_similarity = similarity
            if max_similarity >= threshold:
                break  # Found a sufficiently similar segment, exit early

    return max_similarity >= threshold, max_similarity
