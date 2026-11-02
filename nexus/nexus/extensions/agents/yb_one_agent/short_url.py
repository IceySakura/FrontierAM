"""长链接 → 短链接转换工具。

用于 SFT 数据预处理：把工具返回中的长签名 URL 统一替换为短链，
与工程侧 Go 实现的 GenerateCode 算法对齐（7 位 base62，拒绝采样消除 modulo bias）。

用法::

    from nexus.extensions.agents.yb_one_agent.short_url import ShortUrlMapper

    mapper = ShortUrlMapper(domain="https://yb.agent")
    short = mapper.shorten("https://hunyuan-test-125834...?q-sign-algorithm=sha1&...")
    # => "https://yb.agent/m/A3fB7xQ"

    # 幂等：同一 URL 多次调用返回同一短链
    assert mapper.shorten("https://hunyuan-test-125834...?q-sign-algorithm=sha1&...") == short

    # 反查
    assert mapper.resolve(short) == "https://hunyuan-test-125834...?q-sign-algorithm=sha1&..."

    # 批量替换文本中的所有长 URL
    text = mapper.replace_urls_in_text(raw_text)
"""

from __future__ import annotations

import re
import secrets
import threading


# ── 常量（与 Go 侧 GenerateCode 对齐） ────────────────────────────────────

_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_CODE_LEN = 11
_REJECT_THRESHOLD = 248  # 4 * 62，保证 byte % 62 严格均匀

_DEFAULT_DOMAIN = 'https://yb.agent'
_DEFAULT_MIN_URL_LENGTH = 80

_URL_RE = re.compile(r'https?://\S+')


# ── Code 生成 ──────────────────────────────────────────────────────────────


def generate_code() -> str:
    """生成 7 位 base62 随机短码，与 Go 侧 GenerateCode 算法对齐。

    使用拒绝采样消除 modulo bias：只接受 [0, 248) 范围内的随机字节，
    使 62 个字符严格均匀分布。
    """
    out: list[str] = []
    while len(out) < _CODE_LEN:
        buf = secrets.token_bytes(_CODE_LEN * 2)
        for b in buf:
            if b < _REJECT_THRESHOLD:
                out.append(_ALPHABET[b % 62])
                if len(out) == _CODE_LEN:
                    break
    return ''.join(out)


# ── 映射器 ──────────────────────────────────────────────────────────────────


class ShortUrlMapper:
    """长 URL ↔ 短链双向映射。线程安全，同一 URL 幂等返回同一短链。"""

    def __init__(self, domain: str = _DEFAULT_DOMAIN):
        self._domain = domain.rstrip('/')
        self._lock = threading.Lock()
        self._long_to_short: dict[str, str] = {}
        self._short_to_long: dict[str, str] = {}

    def shorten(self, long_url: str) -> str:
        """将长 URL 映射为短链。同一 URL 多次调用返回相同短链。"""
        with self._lock:
            if long_url in self._long_to_short:
                return self._long_to_short[long_url]

            code = generate_code()
            while code in self._short_to_long:
                code = generate_code()

            short_url = f'{self._domain}/{code}'
            self._long_to_short[long_url] = short_url
            self._short_to_long[short_url] = long_url
            return short_url

    def resolve(self, short_url: str) -> str | None:
        """反查短链对应的长 URL。"""
        with self._lock:
            return self._short_to_long.get(short_url)

    def replace_urls_in_text(
        self,
        text: str,
        min_length: int = _DEFAULT_MIN_URL_LENGTH,
    ) -> str:
        """替换文本中所有超过 min_length 的 http(s) URL 为短链。

        短于阈值的 URL（如短链本身、短域名）不替换。
        本地路径（/data/...）不受影响。
        """

        def _replacer(m: re.Match) -> str:
            url = m.group(0).rstrip('",;>)]\'}')
            trailing = m.group(0)[len(url) :]
            if len(url) < min_length:
                return m.group(0)
            return self.shorten(url) + trailing

        return _URL_RE.sub(_replacer, text)

    def get_mapping(self) -> dict[str, str]:
        """返回 {短链: 长URL} 映射表的副本。"""
        with self._lock:
            return dict(self._short_to_long)

    def __len__(self) -> int:
        with self._lock:
            return len(self._long_to_short)


if __name__ == '__main__':
    mapper = ShortUrlMapper(domain='http://yb.woa.com/')
    short = mapper.shorten(
        'https://hunyuan-test-1258344703.cos.ap-guangzhou.myqcloud.com/text2img/242f6d6987b7db0deade1b4647d6abff/20260515141546_h2_MKZ5_2ed36aa91fd2b906e60f91b6cfe43bc0?q-sign-algorithm=sha1&q-ak=AKIDRl074nOsGdJ9zjMsCRWP3ShmgS3VtX4S&q-sign-time=1778825746;1810361746&q-key-time=1778825746;1810361746&q-header-list=host&q-url-param-list=&q-signature=9be0f33da3e1c05b79c2ce12e875e525b0649ee2'
    )
    print(short)
