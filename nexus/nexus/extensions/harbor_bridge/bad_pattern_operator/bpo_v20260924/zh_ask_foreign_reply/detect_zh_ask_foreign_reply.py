"""zh_ask_foreign_reply — 中文提问，给人看的那条回复却是外文。

检测规则（改行为只改本文件；meta.yaml 不参与判定）
    范围：整条轨迹里每一条 user 都看一遍，不是只扫第一轮或整段末轮。
    每个中文提问最多一条 Finding；turn_index 是被判定的那条 assistant。

    配对：对每条 role=user，窗口到下一条 user 为止。
    提问主文字必须是汉字，否则本问不报（英文脚手架 / task-notification
    等自然被挡住）。
    回复只取窗口里最后一条 tool_calls 为空、且去掉代码后仍有正文的
    assistant。全是工具轮则本问不报。过程稿 I'll start 通常带着工具，不报。

    字段：提问走 WorkBuddy / CodeBuddy 的 <user_query>；没有该标签时剥
    user_info / system-reminder / additional_data 外壳，再看剩下的。
    不看 system、reasoning、tool 参数、```代码块``` 和 `行内代码`。

    判定：去掉代码后，按 Unicode 区间归入文字系统。标点、数字、空格不计入。
    是否够长仍按字符数（提问 ≥8、回复 ≥50），避免短句误报。
    主文字按词计：连续拉丁字母算 1 个词，其余文字每个字算 1 个词。
    这样 isOnWorkbenchOrPortalSubTab 只算 1 个拉丁词，不会压过「修复完成」。
    取词占比最高的一族，且 ≥ DOMINANT_SHARE（60%）：
      - 提问主文字是汉字
      - 回复主文字不是汉字
    若回复主文字是拉丁字母，还要像英文句子：至少 MIN_FUNCTION_WORDS（3）个
    英语虚词（the / is / to / please …）。JSON、bash、型号表几乎没有虚词，不报。
    韩文、假名墙不走虚词：那些字本身就是句子单位。
    → 报。中文问中文答、英文问任何答、中文里夹英文或标识符不报。

要抓的坏样本
    中文提问，最后一条给人看的回复整段大片英文（或韩文、日文假名等）。
不抓
    中文回复里夹英文术语、函数名、包名表，但按词计仍以汉字为主。
    拉丁占比高但没有英文虚词的 JSON / 命令 / 型号表。
    英文提问、英文或中文回答。
    带 tool_calls 的过程稿。

为什么不「识别这是中文还是日语」
    我们没有用语言模型去猜语种。每个字符在 Unicode 里都有一个编号
    （ord('啊') == 0x554A）。相邻编号的字来自同一套文字，叫文字系统
    （script）：汉字一区、拉丁字母一区、日文假名一区、韩文一区。
    下面的 0x3040 这类数就是「这块字表从几号编到几号」，不是魔法。

    用文字系统当近似语种，是因为：
    - 韩文、假名、阿拉伯文几乎等于对应语言，数一下就知道
    - 中文和英文正好落在汉字 vs 拉丁字母两套字里
    - 不需要词典或 n-gram，同事改门槛只改三个常量

    已知限制：中文和日文都用汉字。日文主要靠假名暴露；只有汉字、
    没有假名的日文，会被当成汉字，中文提问下可能漏报。
    拉丁「像句子」用的是英语虚词，不是通用 LID。法德西等整段拉丁、
    但几乎没有 the/is/to 的，可能漏报。用户点名要英文的演讲稿/翻译
    仍会报：规则不判断意图。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .._core.detector import Finding
from .._core.messages import Message, TrajectoryView
from .._core.rule_detector import RuleDetector
from .._core.user_query import extract_user_query

# 提问往往很短（「请解释二叉搜索树」），8 个汉字就够认主语言。
# 回复要够长才算「整段大片」，避免 "OK I will fix it." 这种短句误报。
# 够长按字符计；主语言按词计（拉丁词 vs 汉字），最高族不到 60% 视为夹杂，不判。
MIN_USER_CHARS = 8
MIN_REPLY_CHARS = 50
DOMINANT_SHARE = 0.6
MIN_FUNCTION_WORDS = 3
USER_SCRIPT = "cjk"

# 英语封闭类。句子里密度高；sheetId / conda / E14_7_KUN_AHP 里几乎没有。
# 只当「这段拉丁像不像英语在说话」，不扩展成翻译词典。
_EN_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "in", "that", "it", "for", "on", "with", "as",
    "i", "you", "we", "they", "this", "have", "has", "had", "not", "but",
    "from", "or", "at", "by", "please", "let", "me", "my", "your", "our",
    "will", "would", "can", "could", "should", "do", "does", "did",
    "if", "then", "than", "so", "what", "which", "who", "when", "where",
    "how", "there", "their", "them", "these", "those", "all", "any",
    "no", "because", "while", "after", "before", "about", "into",
    "also", "more", "some", "other", "its", "his", "her", "us",
})

_SCRIPT_LABEL = {
    "cjk": "汉字",
    "latin": "拉丁字母",
    "hiragana_katakana": "日文假名",
    "hangul": "韩文",
    "cyrillic": "西里尔字母",
    "arabic": "阿拉伯文",
    "devanagari": "天城文",
}

# Unicode 把同一套文字排成连续编号。每个区间写成 (起始编号, 结束编号)。
# 先匹配假名/韩文，再匹配汉字：否则「あ」可能被误分。
# 例：ord('あ') == 0x3042，落在平假名 0x3040–0x309F。
_SCRIPT_RANGES: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...] = (
    ("hiragana_katakana", (
        (0x3040, 0x309F),  # 平假名 あいう
        (0x30A0, 0x30FF),  # 片假名 アイウ
        (0x31F0, 0x31FF),  # 片假名扩展
        (0xFF66, 0xFF9D),  # 半角片假名
    )),
    ("hangul", (
        (0x1100, 0x11FF),  # 韩文字母部件
        (0x3130, 0x318F),  # 兼容字母
        (0xAC00, 0xD7AF),  # 完整音节 한글
    )),
    ("cyrillic", (
        (0x0400, 0x04FF),  # 俄文等 Привет
        (0x0500, 0x052F),
    )),
    ("arabic", (
        (0x0600, 0x06FF),  # العربية
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    )),
    ("devanagari", (
        (0x0900, 0x097F),  # 印地语 हिन्दी
    )),
    ("cjk", (
        (0x3400, 0x4DBF),  # 汉字扩展
        (0x4E00, 0x9FFF),  # 常用汉字 中文日本語
        (0xF900, 0xFAFF),  # 兼容汉字
    )),
    ("latin", (
        (0x0041, 0x005A),  # A-Z
        (0x0061, 0x007A),  # a-z
        (0x00C0, 0x024F),  # café, naïve 等带调拉丁字母
    )),
)

_FENCED = re.compile(r"```[\s\S]*?```")
_INLINE = re.compile(r"`[^`]+`")


def strip_code(text: str) -> str:
    if not text:
        return ""
    return _INLINE.sub("", _FENCED.sub("", text))


def script_of(char: str) -> Optional[str]:
    """这个字属于哪套文字？标点、数字、空格返回 None，不参与主语言统计。"""
    code = ord(char)
    for family, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= code <= hi:
                return family
    return None


def script_counts(text: str) -> Dict[str, int]:
    """每个计入字符算 1。只用来判断够不够长，不当主语言。"""
    counts: Dict[str, int] = {}
    for char in text:
        family = script_of(char)
        if family is None:
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def token_counts(text: str) -> Dict[str, int]:
    """主语言计数：连续拉丁字母 = 1 词；其余文字每个字 = 1 词。"""
    counts: Dict[str, int] = {}
    index = 0
    length = len(text or "")
    while index < length:
        family = script_of(text[index])
        if family is None:
            index += 1
            continue
        if family == "latin":
            end = index + 1
            while end < length and script_of(text[end]) == "latin":
                end += 1
            counts["latin"] = counts.get("latin", 0) + 1
            index = end
            continue
        counts[family] = counts.get(family, 0) + 1
        index += 1
    return counts


def script_shares(text: str) -> Dict[str, float]:
    counts = token_counts(text)
    total = sum(counts.values())
    if not total:
        return {}
    return {family: round(count / total, 4) for family, count in counts.items()}


def latin_words(text: str) -> List[str]:
    """连续拉丁字母片段，小写。I'll → i + ll。"""
    words: List[str] = []
    index = 0
    length = len(text or "")
    while index < length:
        if script_of(text[index]) != "latin":
            index += 1
            continue
        end = index + 1
        while end < length and script_of(text[end]) == "latin":
            end += 1
        words.append(text[index:end].lower())
        index = end
    return words


def latin_function_word_count(text: str) -> int:
    return sum(1 for word in latin_words(text) if word in _EN_FUNCTION_WORDS)


def dominant_script(text: str, *, min_chars: int) -> Optional[str]:
    """这段话的主文字系统。字太少、或没有一族过 60%，视为无法判定。"""
    if sum(script_counts(text).values()) < min_chars:
        return None
    counts = token_counts(text)
    total = sum(counts.values())
    if not total:
        return None
    family, count = max(counts.items(), key=lambda item: item[1])
    if count / total < DOMINANT_SHARE:
        return None
    return family


def _last_user_facing_assistant(
    messages: List[Message],
    start: int,
) -> Optional[Tuple[int, str]]:
    """窗口内最后一条无 tool_calls、且去掉代码后仍有正文的 assistant。"""
    last: Optional[Tuple[int, str]] = None
    for index in range(start, len(messages)):
        message = messages[index]
        if message.role == "user":
            break
        if message.role != "assistant":
            continue
        if message.tool_calls:
            continue
        reply = strip_code(message.content or "")
        if not reply.strip():
            continue
        last = (index, reply)
    return last


def _snippet(text: str, max_len: int = 200) -> str:
    rendered = (text or "").replace("\n", "\\n")
    return rendered[:max_len] + (" …" if len(rendered) > max_len else "")


class ZhAskForeignReplyDetector(RuleDetector):
    detector_id = "zh_ask_foreign_reply"
    thresholds = {
        "min_user_chars": MIN_USER_CHARS,
        "min_reply_chars": MIN_REPLY_CHARS,
        "dominant_share": DOMINANT_SHARE,
        "min_function_words": MIN_FUNCTION_WORDS,
    }

    def scan(self, view: TrajectoryView) -> List[Finding]:
        findings: List[Finding] = []
        messages = view.messages
        for user_index, user in enumerate(messages):
            if user.role != "user":
                continue
            user_text = strip_code(extract_user_query(user.content or ""))
            user_script = dominant_script(user_text, min_chars=MIN_USER_CHARS)
            if user_script != USER_SCRIPT:
                continue
            picked = _last_user_facing_assistant(messages, user_index + 1)
            if picked is None:
                continue
            index, reply = picked
            reply_script = dominant_script(reply, min_chars=MIN_REPLY_CHARS)
            if not reply_script or reply_script == USER_SCRIPT:
                continue
            function_words = latin_function_word_count(reply)
            if reply_script == "latin" and function_words < MIN_FUNCTION_WORDS:
                continue
            findings.append(Finding(
                rule_id=self.detector_id,
                severity="medium",
                is_issue=True,
                turn_index=index,
                description=(
                    f"assistant 第 {index} 条消息:中文提问外文作答:"
                    f"提问以汉字为主，"
                    f"回复整段以{_SCRIPT_LABEL.get(reply_script, reply_script)}为主"
                    f"（≥{DOMINANT_SHARE:.0%}）"
                ),
                detail={
                    "user_script": user_script,
                    "reply_script": reply_script,
                    "user_script_shares": script_shares(user_text),
                    "reply_script_shares": script_shares(reply),
                    "latin_function_words": function_words,
                    "min_user_chars": MIN_USER_CHARS,
                    "min_reply_chars": MIN_REPLY_CHARS,
                    "dominant_share": DOMINANT_SHARE,
                    "min_function_words": MIN_FUNCTION_WORDS,
                },
                quotes=[{
                    "kind": self.detector_id,
                    "turn_index": index,
                    "role": "assistant",
                    "user_script": user_script,
                    "reply_script": reply_script,
                    "snippet": _snippet(reply),
                }],
            ))
        return findings


__all__ = [
    "ZhAskForeignReplyDetector",
    "dominant_script",
    "latin_function_word_count",
    "latin_words",
    "script_counts",
    "script_of",
    "script_shares",
    "strip_code",
    "token_counts",
]
