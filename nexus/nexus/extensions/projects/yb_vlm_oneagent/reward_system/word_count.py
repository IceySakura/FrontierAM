"""Markdown 文本字数统计模块。

整合两步流水线：
1. markdown_to_visible_text：将 Markdown 文本转为渲染后可见的纯文本（去除格式标记）
2. count_world_len：统计可见文本的字数（中文字 + 英文单词 + 其他非空白字符）

使用方式：
    from word_count import markdown_word_count, count_world_len, markdown_to_visible_text

    # 一站式：Markdown → 字数
    word_count = markdown_word_count(markdown_text)

    # 分步使用
    visible = markdown_to_visible_text(markdown_text)
    word_count = count_world_len(visible)
"""

import re
import string

import markdown  # pyright: ignore[reportMissingImports]

from bs4 import BeautifulSoup, NavigableString  # pyright: ignore[reportMissingImports]


__all__ = [
    'markdown_to_visible_text',
    'count_words_and_chars',
    'count_words_and_chars_all',
    'count_world_len',
    'markdown_word_count',
]

# ============================================================
# Part 1: Markdown → 可见纯文本
# ============================================================

_MD_EXTENSIONS = ['extra', 'sane_lists']
_BLOCK_TAGS = (
    'p',
    'div',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'blockquote',
    'pre',
    'li',
    'hr',
    'th',
    'td',
)
_CODE_ANCESTORS = ('code', 'pre')
_STRUCTURAL_WHITESPACE_PARENTS = ('ol', 'ul', 'li', 'table', 'thead', 'tbody', 'tr')
_STRIKE_RE = re.compile(r'~~(.+?)~~')
_MULTI_BLANK_RE = re.compile(r'\n{3,}')


def markdown_to_visible_text(text):
    """将 Markdown 文本转换为渲染后人眼真正可见的纯文本。

    口径约定：
    - 标记符号去掉、文字保留：标题 #、强调 * _ ** ~~、引用 >、表格 | 与分隔行、
      水平线 --- ***、HTML 标签与注释、转义符 \\。
    - 链接 [文本](url)：保留「文本」，去掉 URL；裸 URL / 自动链接保留其显示文本。
    - 图片 ![alt](url)：alt 不算可见字（图片渲染成图，alt 通常不显示）。
    - 代码块 ``` ``` 与行内代码 `code`：内容原样保留、计入字数。
    - 列表：列表项正文计入；有序列表还原序号「1. 2. ...」、无序列表还原项目符号「• 」。
    - 数学公式 $...$、$$...$$ 的定界符不处理、原样保留。
    """
    if not isinstance(text, str):
        raise TypeError(f'text must be str, got {type(text).__name__}: {text!r}')

    html = markdown.markdown(text, extensions=_MD_EXTENSIONS)
    soup = BeautifulSoup(html, 'html.parser')

    # 移除图片（alt 不算可见字）
    for img in soup.find_all('img'):
        img.decompose()

    # 处理删除线（python-markdown 不解析 ~~，标记会原样留在文本里）
    # 只处理非代码区域，代码块/行内代码里的内容保持原样
    for node in list(soup.find_all(string=True)):
        if any(parent.name in _CODE_ANCESTORS for parent in node.parents):
            continue
        new_text = _STRIKE_RE.sub(r'\1', node)
        if new_text != node:
            node.replace_with(new_text)

    # 还原有序列表序号
    for ol in soup.find_all('ol'):
        start = ol.get('start')
        index = int(start) if isinstance(start, str) and start else 1
        for li in ol.find_all('li', recursive=False):
            li.insert(0, NavigableString(f'{index}. '))
            index += 1

    # 还原无序列表项目符号
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li', recursive=False):
            li.insert(0, NavigableString('\u2022 '))

    # 清理结构性空白（HTML 标签间的纯空白文本节点）
    for node in list(soup.find_all(string=True)):
        if node.parent is not None and node.parent.name in _STRUCTURAL_WHITESPACE_PARENTS and not node.strip():
            node.extract()

    # 嵌套列表前补换行，避免父列表项文本与子列表首项粘连
    for nested_list in soup.find_all(['ul', 'ol']):
        nested_list.insert_before(NavigableString('\n'))

    # <br> 换成单个换行，并吸收其后的源码软换行避免多出空行
    for br in soup.find_all('br'):
        nxt = br.next_sibling
        if isinstance(nxt, NavigableString) and nxt[:1] == '\n':
            nxt.replace_with(nxt[1:])
        br.replace_with('\n')

    # 块级元素之间补换行，避免相邻块文本粘连
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append('\n')

    visible = soup.get_text()
    return _MULTI_BLANK_RE.sub('\n\n', visible).strip()


# ============================================================
# Part 2: 字数统计
# ============================================================


def count_words_and_chars(input_string):
    """基础统计：英文单词数 + 中文字符数。

    - 英文单词：连续的字母+数字序列算 1 个词
    - 中文字符：每个汉字算 1 字
    - 标点符号、特殊字符等不计入
    """
    words = re.findall(r'[a-zA-Z\d]+', input_string)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', input_string)
    return len(words) + len(chinese_chars)


def count_words_and_chars_all(input_string):
    """完整统计：英文单词数 + 中文字符数 + 其他非空白字符数。

    在基础统计之上，额外计入无法被英文/中文匹配到的非空白字符
    （如日文假名、emoji、特殊符号等）。
    """
    words_chars = string.ascii_letters + string.digits + string.punctuation
    words = re.findall(rf'[{words_chars}]+', input_string)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', input_string)

    # 计算已匹配的总字符数
    words_str = ''.join(words)
    chinese_str = ''.join(chinese_chars)
    total_matched_chars = len(words_str) + len(chinese_str)

    # 计算其他非空白字符的数量
    other_chars_count = len(re.sub(rf'[{string.whitespace}]', '', input_string)) - total_matched_chars

    return len(words) + len(chinese_chars) + other_chars_count


def count_world_len(input_string):
    """统计文本字数。

    策略：
    - 短文本（基础统计 ≤ 30字）：只算中文字 + 英文词，忽略标点符号
    - 长文本（基础统计 > 30字）：额外加上其他非空白字符
    """
    # raw_count = count_words_and_chars(input_string)
    # if raw_count <= 30:
    #     return raw_count
    return count_words_and_chars_all(input_string)


# ============================================================
# Part 3: 一站式接口
# ============================================================


def markdown_word_count(text):
    """一站式接口：Markdown 文本 → 可见字数。

    先将 Markdown 转为可见纯文本（去除格式标记），再统计字数。
    """
    visible = markdown_to_visible_text(text)
    return count_world_len(visible)


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    doc = r"""开头先写一段正常的说明文字，作为整篇的引子。

# 标题
这是**加粗**、*斜体*、~~删除线~~和`行内代码`，都是常见的行内格式。

正文里经常引用链接，比如看[百度](https://baidu.com)，再配一张图片![一只橘猫](http://x/cat.png)（图片 alt 不算可见字）。裸链接 <http://x> 和 http://x 也会原样保留。

下面用一个有序列表说明几个步骤：

1. 第一
2. 第二
    - 子项A
    - 子项B
3. 第三

展示代码时会放进代码块，里面内容全部原样保留、不解析：

```python
# 注释 **不加粗** [非链接](x) a~~b~~c
print(1)
```

数据一般用表格呈现：

| 姓名 | 年龄 |
| --- | --- |
| 张三 | 18 |

涉及公式要和货币符号区分：块公式 $$E=mc^2$$，而货币 $100 和 $200 应当保留。

> 引用内容

---

转义\*星号也要还原，再夹一个 <b>HTML</b> 标签。"""

    visible = markdown_to_visible_text(doc)
    word_count = count_world_len(visible)

    print('===== 原始 Markdown =====')
    print(doc)
    print('\n===== 可见文本 =====')
    print(visible)
    print('\n===== 字数统计 =====')
    print(f'可见字数: {word_count}')
    no_space = re.sub(r'\s', '', visible)
    print(f'纯字符数(去空白): {len(no_space)}')
