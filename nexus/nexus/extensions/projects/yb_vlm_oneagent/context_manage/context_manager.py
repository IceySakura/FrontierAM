"""
context_manager: yb_vlm_agent oneagent 协议的上下文管理实现。

按 role 设定 max-token 配额：
  - system_prompt           : 4k tokens（仅文本）
  - 历史 user prompt         : 3k tokens（图+文，先文本后图片）
  - 历史 assistant answer    : 1k tokens（仅文本，截断）
  - 当前轮 tool round 预算   : 18.5k tokens（每轮工具调用+工具返回合计）
      * 每轮内：先扣除模型 tool_call 文本与图搜返回图片 token，剩余分配给文搜返回文本

另按 round 数限制历史长度：
  - 仅保留最近 N 轮 (q, a, imgs)；更早的轮次直接丢弃，不参与 token 预算
    （先按轮裁剪、再按 token 截断）。N 由环境变量 ``YB_VLM_MAX_HISTORY_ROUNDS``
    指定，未设置时直接报错（无默认值，避免静默偏差）。

主要入口：
  - build_managed_messages: 在进入 agent rollout 前调用，构造经过截断/resize 后的初始
    messages（system + history + 当前 user）。同时返回 user/tool 图片各自的
    (min_pixels, max_pixels) 配额，用于后续工具返回图片的 base64 编码。
  - apply_tool_round_budget: 在每一次工具调用后调用，根据工具调用文本与图搜返回图片
    实际 token 占用，计算文搜文档允许的 max_tokens，并在必要时按 doc 粒度截断文搜返回。
"""

import os
import re

from dataclasses import dataclass, field
from typing import Any, Optional

from .image_resizer import (
    interleave_resize,
)
from .token_counter import (
    TokenCounter,
    get_image_size,
)


# ─────────── 默认预算（可按需调整） ───────────
DEFAULT_SYSTEM_TOKENS = 4 * 1024  # 4k
DEFAULT_HISTORY_USER_TOKENS = 3 * 1024  # 3k（含文本+图片）
DEFAULT_HISTORY_USER_TEXT_TOKENS = 1 * 1024  # user prompt 文本部分上限 1k
DEFAULT_HISTORY_ANS_TOKENS = 1 * 1024  # 1k
DEFAULT_TOOL_ROUND_TOKENS = int(18.5 * 1024)  # 18.5k
# 单轮 tool round 中协议级 special token 的保守预留（chat-template 包裹 / 分隔等）
DEFAULT_TOOL_ROUND_SPECIAL_RESERVE_TOKENS = 256


# 历史轮数上限：仅保留最近 N 轮 (q, a, imgs)，更早的轮次直接丢弃。
# 强制走环境变量 ``YB_VLM_MAX_HISTORY_ROUNDS``——这是 per-run 全局配置，
# 在 dataset / rollout 等多入口都需要保持一致；缺失或非法则直接报错，避免
# 静默回落到不同的默认值导致 token 统计漂移。
_MAX_HISTORY_ROUNDS_ENV = 'YB_VLM_MAX_HISTORY_ROUNDS'


def _resolve_max_history_rounds() -> int:
    raw = os.environ[_MAX_HISTORY_ROUNDS_ENV]
    n = int(raw)
    if n < 0:
        raise ValueError(f'{_MAX_HISTORY_ROUNDS_ENV} must be a non-negative integer, got {raw!r}')
    return n


@dataclass
class ContextBudget:
    """上下文预算配置。"""

    system_max_tokens: int = DEFAULT_SYSTEM_TOKENS
    history_user_max_tokens: int = DEFAULT_HISTORY_USER_TOKENS
    # user prompt 文本部分硬上限（避免文本占满 3k 后图片无预算）
    history_user_text_max_tokens: int = DEFAULT_HISTORY_USER_TEXT_TOKENS
    history_answer_max_tokens: int = DEFAULT_HISTORY_ANS_TOKENS
    tool_round_max_tokens: int = DEFAULT_TOOL_ROUND_TOKENS
    # 单轮 tool round 中协议级 special token 的保守预留
    # （role=tool 的 chat-template 包裹 / 各 content block 分隔符等，与具体业务无关）
    tool_round_special_reserve_tokens: int = DEFAULT_TOOL_ROUND_SPECIAL_RESERVE_TOKENS

    # 图片 resize 相关
    factor: int = 32
    # 协议约定：所有图片（user / tool）传入 resize_smart 的 min_pixels 都固定为 256*256
    min_pixels: int = 256 * 256  # 单图最小像素（resize_smart 下限）
    base_max_pixels: int = 2048 * 2048  # 用户图片最大像素上限
    tool_max_pixels: int = 1024 * 1024  # 工具返回图片最大像素上限
    # 多图退化阈值：若按预算分到的单图像素低于此值则退化为 interleave 最小分辨率
    multi_min_pixels: int = 256 * 256
    interleave_max_size: int = 256
    multi_nums_limit: int = 40
    max_ratio: int = 200


# ───────────────────────────────────────────────
# 图片 resize 计划
# ───────────────────────────────────────────────
@dataclass
class ImageMeta:
    """一张用户图片的原始来源与尺寸。"""

    source: Any
    width: int
    height: int


@dataclass
class ImageResizePlan:
    """对一张图片的 resize 决策（min/max pixels）。"""

    min_pixels: int
    max_pixels: int
    # 期望 resize 后的尺寸（用于估算 token，可选）
    new_size: tuple[int, int] | None = None
    estimated_tokens: int = 0


@dataclass
class ManagedMessages:
    """build_managed_messages 的返回结果。"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    # 当前轮（最新一轮）用户图片 resize 决策（按 images 列表顺序）
    cur_user_image_plans: list[ImageResizePlan] = field(default_factory=list)
    # 工具返回图片的 (min_pixels, max_pixels)，由 ContextManager 按预算固定下来
    tool_image_min_pixels: int = 0
    tool_image_max_pixels: int = 0
    # 已使用的 token 数（system + history + 当前 user）
    used_tokens: int = 0


# ───────────────────────────────────────────────
# 主类
# ───────────────────────────────────────────────
class ContextManager:
    """按上下文管理协议组织 messages 并管理 token 预算。"""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        token_counter: TokenCounter | None = None,
        think_token: str = '',
    ):
        self.budget = budget or ContextBudget()
        if token_counter is None:
            raise ValueError(
                'ContextManager requires an explicit token_counter; construct '
                'TokenCounter(tokenizer) and pass it in (no global fallback).'
            )
        self.tc = token_counter
        # 与外层 think_token 对齐
        self.think_token = think_token

    # ───────── system prompt ─────────
    def build_system_message(self, system_prompt: str) -> tuple[dict[str, Any], int]:
        text, tokens = self.tc.truncate_text_by_tokens(system_prompt or '', self.budget.system_max_tokens)
        return {'role': 'system', 'content': text}, tokens

    # ───────── 历史 assistant answer ─────────
    def truncate_history_answer(self, answer: str) -> tuple[str, int]:
        return self.tc.truncate_text_by_tokens(answer or '', self.budget.history_answer_max_tokens)

    # ───────── 计算用户消息中图片的 resize 计划 ─────────
    def _load_user_image_metas(self, image_sources: list[Any]) -> list[ImageMeta]:
        """在上下文管理开始时读取用户图片尺寸，并保留原始图片来源。"""
        metas: list[ImageMeta] = []
        for idx, source in enumerate(image_sources):
            try:
                w, h = get_image_size(source)
            except Exception as exc:
                raise ValueError(f'Failed to read user image size at index {idx}: {source!r}') from exc
            metas.append(ImageMeta(source=source, width=w, height=h))
        return metas

    def _plan_user_images(self, image_metas: list[ImageMeta], remain_tokens_for_images: int) -> list[ImageResizePlan]:
        """对一条 user message 中的图片计算 resize 方案：
        1. 先用 interleave_resize 根据图片数量与剩余 token 计算单图 max_pixels；
        2. 再基于已读取的原图尺寸，按 resize_smart 估算实际 token；
        3. 若总 token 仍超出预算，按照比例进一步压缩 max_pixels（最低 256x256）。
        """
        n = len(image_metas)
        if n == 0:
            return []

        # 多图分配
        min_pixels, max_pixels = interleave_resize(
            image_count=n,
            reserved_pixel_tokens=remain_tokens_for_images,
            factor=self.budget.factor,
            base_max_pixels=self.budget.base_max_pixels,
            base_min_pixels=self.budget.min_pixels,
            multi_min_pixels=self.budget.multi_min_pixels,
            interleave_max_size=self.budget.interleave_max_size,
            multi_nums_limit=self.budget.multi_nums_limit,
        )

        plans: list[ImageResizePlan] = []
        for meta in image_metas:
            tokens, new_size = self.tc.estimate_image_tokens_after_resize(
                width=meta.width,
                height=meta.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_ratio=self.budget.max_ratio,
            )
            plans.append(
                ImageResizePlan(
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    new_size=new_size,
                    estimated_tokens=tokens,
                )
            )

        # 若 token 超预算，则降级 max_pixels 重新计算
        total = sum(pl.estimated_tokens for pl in plans)
        if total > remain_tokens_for_images > 0:
            shrink_max_pixels = max(
                self.budget.interleave_max_size * self.budget.interleave_max_size,
                int(max_pixels * (remain_tokens_for_images / max(total, 1))),
            )
            plans = []
            for meta in image_metas:
                tokens, new_size = self.tc.estimate_image_tokens_after_resize(
                    width=meta.width,
                    height=meta.height,
                    min_pixels=min(min_pixels, shrink_max_pixels),
                    max_pixels=shrink_max_pixels,
                    max_ratio=self.budget.max_ratio,
                )
                plans.append(
                    ImageResizePlan(
                        min_pixels=min(min_pixels, shrink_max_pixels),
                        max_pixels=shrink_max_pixels,
                        new_size=new_size,
                        estimated_tokens=tokens,
                    )
                )
        return plans

    def _build_internal_user_image_content(
        self, image_metas: list[ImageMeta], plans: list[ImageResizePlan]
    ) -> list[dict[str, Any]]:
        """构造内部图片 content：保留 source/原始尺寸/resize 决策，暂不编码 base64。"""
        content: list[dict[str, Any]] = []
        for meta, plan in zip(image_metas, plans, strict=False):
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': meta.source if isinstance(meta.source, str) else '',
                    'detail': 'high',
                    '_image_source': meta.source,
                    '_image_meta': {
                        'width': meta.width,
                        'height': meta.height,
                    },
                    '_resize_plan': {
                        'min_pixels': plan.min_pixels,
                        'max_pixels': plan.max_pixels,
                        'new_size': plan.new_size,
                        'estimated_tokens': plan.estimated_tokens,
                    },
                },
            })
        return content

    # ───────── 历史 user 消息 ─────────
    def build_history_user_message(
        self, query: str, image_sources: list[Any]
    ) -> tuple[dict[str, Any], list[ImageResizePlan], int]:
        """构造一条历史 user message，并返回它的图片 resize 计划与 token 使用情况。

        协议预算分配：
          history_user_max_tokens = 文本 token + 图片 token（已含 patch + 标记）
          - 文本部分硬上限 history_user_text_max_tokens（默认 1k），超过即截断；
          - 图片预留 token = history_user_max_tokens - 文本实际 token；
            * 文本未达 1k：图片预留 = 3k - 实际文本 token；
            * 文本达到 1k 上限（被截断）：图片预留 = 3k - 1k = 2k。
        """
        text_budget = self.budget.history_user_text_max_tokens
        text, text_tokens = self.tc.truncate_text_by_tokens(query or '', text_budget)
        remain_for_images = max(0, self.budget.history_user_max_tokens - text_tokens)

        image_metas = self._load_user_image_metas(image_sources)
        plans = self._plan_user_images(image_metas, remain_for_images)
        img_tokens = sum(pl.estimated_tokens for pl in plans)

        # 组装内部 message content：记录原始尺寸和 resize 决策，出口再统一转 base64。
        content = self._build_internal_user_image_content(image_metas, plans)
        if text:
            content.append({'type': 'text', 'text': text})

        msg = {'role': 'user', 'content': content}
        return msg, plans, text_tokens + img_tokens

    # ───────── 历史 assistant 消息 ─────────
    def build_history_assistant_message(self, answer: str) -> tuple[dict[str, Any], int]:
        text, tokens = self.truncate_history_answer(answer)
        # 与原逻辑保持一致：在末尾补 think_token
        if self.think_token:
            text = text + self.think_token
        return {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': text}],
        }, tokens

    # ───────── 当前轮 user 消息 ─────────
    def build_current_user_message(
        self, query: str, image_sources: list[Any]
    ) -> tuple[dict[str, Any], list[ImageResizePlan], int]:
        """当前轮 user 消息：与历史同口径，文本受 history_user_text_max_tokens（默认 1k）
        硬上限约束（超过即截断），剩余预算按 (3k - 实际文本 token) 给图片。

        说明：think_token 只是开关占位符（极短，几乎不影响 token），统一拼接到文本末尾后
        再做截断与统计，避免被丢弃。
        """
        raw_text = (query or '') + (self.think_token or '')
        text_budget = self.budget.history_user_text_max_tokens
        text, text_tokens = self.tc.truncate_text_by_tokens(raw_text, text_budget)
        remain_for_images = max(0, self.budget.history_user_max_tokens - text_tokens)

        image_metas = self._load_user_image_metas(image_sources)
        plans = self._plan_user_images(image_metas, remain_for_images)
        img_tokens = sum(pl.estimated_tokens for pl in plans)

        content = self._build_internal_user_image_content(image_metas, plans)
        content.append({'type': 'text', 'text': text})
        return {'role': 'user', 'content': content}, plans, text_tokens + img_tokens


# ───────────────────────────────────────────────
# 入口便捷函数
# ───────────────────────────────────────────────
def build_managed_messages(
    system_prompt: str,
    history_queries: list[str],
    history_answers: list[str],
    history_images: list[list[Any]],
    cur_query: str,
    cur_images: list[Any],
    think_token: str = '',
    budget: ContextBudget | None = None,
    token_counter: TokenCounter | None = None,
) -> ManagedMessages:
    """按上下文管理协议构造初始 messages 列表。

    Returns:
        ManagedMessages，包含：
        - messages: 用户图片保留原始 URL / 本地路径，并在
          ``image_url._resize_plan.new_size`` 中携带最终 resize 尺寸。
        - cur_user_image_plans: 最新一轮用户图片的 resize 计划。
        - tool_image_min_pixels / tool_image_max_pixels: 工具返回图片的固定 resize 配额。
        - used_tokens: 已使用的 token 数（system + history + 当前 user）。
    """
    cm = ContextManager(budget=budget, token_counter=token_counter, think_token=think_token)
    used = 0
    messages: list[dict[str, Any]] = []

    # 1) system
    sys_msg, sys_tokens = cm.build_system_message(system_prompt)
    messages.append(sys_msg)
    used += sys_tokens

    # 2) history rounds
    history_queries = history_queries or []
    history_answers = history_answers or []
    history_images = history_images or []
    if not (len(history_queries) == len(history_answers) == len(history_images)):
        raise ValueError(
            f'history 长度不一致 q={len(history_queries)} a={len(history_answers)} img={len(history_images)}'
        )

    # 按轮裁剪：仅保留最近 N 轮（先按轮裁、再按 token 截断）。
    # N 直接来自环境变量 ``YB_VLM_MAX_HISTORY_ROUNDS``（单一来源），
    # 不暴露为 ContextBudget 字段，避免出现两份配置面而漂移。
    _keep = _resolve_max_history_rounds()
    if _keep == 0:
        history_queries = []
        history_answers = []
        history_images = []
    elif len(history_queries) > _keep:
        history_queries = history_queries[-_keep:]
        history_answers = history_answers[-_keep:]
        history_images = history_images[-_keep:]

    for q, a, imgs in zip(history_queries, history_answers, history_images, strict=False):
        u_msg, _, u_tokens = cm.build_history_user_message(q, list(imgs or []))
        a_msg, a_tokens = cm.build_history_assistant_message(a)
        messages.append(u_msg)
        messages.append(a_msg)
        used += u_tokens + a_tokens

    # 3) 当前轮 user
    cur_msg, cur_plans, cur_tokens = cm.build_current_user_message(cur_query, cur_images)
    messages.append(cur_msg)
    used += cur_tokens

    # 4) 工具返回图片的预算：对齐多图分配（不超过 tool_max_pixels）
    tool_min_pixels = cm.budget.min_pixels
    tool_max_pixels = cm.budget.tool_max_pixels

    return ManagedMessages(
        messages=messages,
        cur_user_image_plans=cur_plans,
        tool_image_min_pixels=tool_min_pixels,
        tool_image_max_pixels=tool_max_pixels,
        used_tokens=used,
    )


# ───────────────────────────────────────────────
# web_search res_str 按 doc 边界切分
# ───────────────────────────────────────────────
# 匹配 oneagent 协议 web_search 工具返回的
#   [web_doc N begin] ... [web_doc N end]
# 段（按整段 begin/end 配对，跨行匹配）。
_DEFAULT_WEB_DOC_PATTERN = re.compile(r'\[web_doc \d+ begin\][\s\S]*?\[web_doc \d+ end\]')


def split_web_doc_segments(res_str: str, pattern: Optional['re.Pattern'] = None) -> list[str]:
    """把 web_search 接口返回的 res_str 按 doc 边界切成 doc 列表。

    - 切分后每个元素都是一篇完整 doc 的规范化文本（含 [web_doc N begin]/[web_doc N end] 包裹）；
    - doc 之间的 "\n" 分隔符会在切分时丢弃，外层重新拼接时仍按 "\n" join 即可还原；
    - 调用方可自定义 pattern。
    """
    if not res_str:
        return []
    pat = pattern if pattern is not None else _DEFAULT_WEB_DOC_PATTERN
    return pat.findall(res_str)


def apply_tool_round_budget(
    tool_call_text: str,
    web_doc_segments: list[list[str]] | None = None,
    fixed_extra_text: str = '',
    special_reserve_tokens: int = 0,
    tool_round_max_tokens: int = DEFAULT_TOOL_ROUND_TOKENS,
    image_tokens: int = 0,
    token_counter: TokenCounter | None = None,
) -> dict[str, Any]:
    """对一轮工具调用做 token 预算控制。

    一轮 tool round = 模型输出的 tool_call 文本 + 工具返回（图搜图片 + 图搜返回 list 中的文本
    + 文搜返回文本）+ role=tool 协议级 special token。预留 18.5k：
        web_max_tokens = tool_round_max_tokens
                       - tool_call_tokens          (模型本轮原始输出，含 <think>/<tool_calls>)
                       - image_tokens              (图搜返回图片，由调用方按 resize 后尺寸算好)
                       - fixed_extra_tokens        (prefix_text + vision_search 返回的 list 中
                                                    的 str 元素，已固定要进 prefill)
                       - special_reserve_tokens    (role=tool 的 chat-template 包裹常量预留)

    Args:
        tool_call_text: 模型本轮原始输出文本（含所有 <think>/<tool_calls>/<tool_call> 内容）。
        web_doc_segments: 每次 web_search 调用对应的 doc 列表，外形
            ``[[doc0_call0, doc1_call0, ...], [doc0_call1, ...], ...]``。
            按 doc 粒度计算 token 累计：第一个让累计超过 web_max_tokens 的 doc 整篇丢弃，
            该 doc 之后的所有 doc 也一并丢弃（保留下来的 doc 会按原顺序回到对应的 call 内）。
        fixed_extra_text: 已固定要进 prefill 的额外文本（不可截断），用于扣减预算。
        special_reserve_tokens: role=tool 的协议级 special token 预留量。
        tool_round_max_tokens: 单轮工具预算（默认 18.5k）。
        image_tokens: 本轮图搜返回图片的合计 token 数，**由调用方按 resize 后的真实尺寸算好**
            （见 ``yb_vision_search``：下载/resize 后用 ``count_image_tokens(w, h)`` 累加）。
            本函数不再从 path/URL 反推尺寸——对 ``data:image/...;base64,...`` 这种已内联的
            图片反解既不可靠也无必要。无图搜时传 0。

    Returns:
        dict 包含：
          tool_call_tokens:        模型本轮输出文本 token 数
          image_tokens:            图搜返回图片合计 token 数（原样回显入参）
          extra_fixed_tokens:      fixed_extra_text 占用 token 数
          special_reserve_tokens:  传入的 special token 预留
          web_max_tokens:          留给文搜返回文本的最大 token 数
          truncated_web_segments:  按 doc 粒度截断后的 doc 列表（与传入 web_doc_segments
                                   形状一致；被丢弃的 doc 不在列表里），未传入则为 None
          kept_doc_count / dropped_doc_count: doc 粒度截断的命中统计
    """
    # token_counter 必须由调用方显式传入。
    if token_counter is None:
        raise ValueError(
            'apply_tool_round_budget requires an explicit token_counter; '
            "the oneagent hook seeds it via ctx.tool_state['token_counter']."
        )
    tc = token_counter

    tool_call_tokens = tc.count_text(tool_call_text or '')
    extra_fixed_tokens = tc.count_text(fixed_extra_text or '')
    special_reserve_tokens = max(0, int(special_reserve_tokens or 0))
    image_tokens = max(0, int(image_tokens or 0))

    # 文搜剩余预算
    web_max_tokens = max(
        0,
        tool_round_max_tokens - tool_call_tokens - image_tokens - extra_fixed_tokens - special_reserve_tokens,
    )

    truncated_web_segments: list[list[str]] | None = None
    kept_doc_count = 0
    dropped_doc_count = 0

    if web_doc_segments is not None:
        # 按 doc 粒度累计 token：第一篇 token 累计超出 web_max_tokens 的 doc 整篇丢弃，
        # 该 doc 之后的所有 doc 也一并丢弃（不在 doc 中段截断、不破坏 doc/url 结构）。
        truncated_web_segments = []
        accum = 0
        stopped = False
        for call_docs in web_doc_segments:
            kept_for_call: list[str] = []
            if not stopped:
                docs = call_docs or []
                for idx, doc in enumerate(docs):
                    if not doc:
                        continue
                    doc_tokens = tc.count_text(doc)
                    if accum + doc_tokens > web_max_tokens:
                        stopped = True
                        dropped_doc_count += sum(1 for d in docs[idx:] if d)
                        # 当前 doc 与本次 call 中其余 doc 全部丢弃
                        # （后续 call 也会因 stopped=True 全部丢弃）
                        break
                    accum += doc_tokens
                    kept_for_call.append(doc)
                    kept_doc_count += 1
            else:
                # 已停止：本次 call 剩余 doc 全部计入 dropped 统计
                dropped_doc_count += sum(1 for d in (call_docs or []) if d)
            truncated_web_segments.append(kept_for_call)

    return {
        'tool_call_tokens': tool_call_tokens,
        'image_tokens': image_tokens,
        'extra_fixed_tokens': extra_fixed_tokens,
        'special_reserve_tokens': special_reserve_tokens,
        'web_max_tokens': web_max_tokens,
        'truncated_web_segments': truncated_web_segments,
        'kept_doc_count': kept_doc_count,
        'dropped_doc_count': dropped_doc_count,
    }
