"The 12 PPT/slide tools for the ``ppt_topic`` task, executing in the sandbox."

from __future__ import annotations

from typing import Any


PPT12_TOOLS: tuple[str, ...] = (
    'visual/ppt_page_info',
    'visual/ppt_text_blocks',
    'visual/ppt_sample_colour',
    'visual/ppt_region_stats',
    'visual/ppt_find_edges',
    'visual/ppt_identify_font',
    'visual/ppt_estimate_font_size',
    'visual/ppt_inspect_region',
    'visual/ppt_overlay_grid',
    'visual/ppt_compare_render',
    'visual/ppt_render_slide',
    'visual/ppt_code_interpreter',
)

PPT12_ALIAS_TOOLS: tuple[str, ...] = ('visual/ppt_run_python',)

PPT12_ALL_REGISTRY_NAMES: tuple[str, ...] = PPT12_TOOLS + PPT12_ALIAS_TOOLS

PPT12_MODEL_NAMES: tuple[str, ...] = (
    'page_info',
    'text_blocks',
    'sample_colour',
    'region_stats',
    'find_edges',
    'identify_font',
    'estimate_font_size',
    'inspect_region',
    'overlay_grid',
    'compare_render',
    'render_slide',
    'code_interpreter',
)

PPT12_MODULES: dict[str, tuple[str, ...]] = {
    'measure_tools': (
        'visual/ppt_page_info',
        'visual/ppt_text_blocks',
        'visual/ppt_sample_colour',
        'visual/ppt_region_stats',
        'visual/ppt_find_edges',
    ),
    'font_tools': ('visual/ppt_identify_font', 'visual/ppt_estimate_font_size'),
    'image_tools': (
        'visual/ppt_inspect_region',
        'visual/ppt_overlay_grid',
        'visual/ppt_compare_render',
    ),
    'compile_tools': ('visual/ppt_render_slide',),
    'code_tool': ('visual/ppt_code_interpreter', 'visual/ppt_run_python'),
}

PPT12_PRESET = 'ppt12'

IMAGE_STATE_KEY = 'image'
CODE_WORKDIR_STATE_KEY = 'pptx_code_workdir'


def expand_agent_tools(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for name in names:
        if name == PPT12_PRESET:
            expanded.extend(PPT12_TOOLS)
        else:
            expanded.append(name)
    return expanded


def load_ppt12(names: tuple[str, ...] = PPT12_ALL_REGISTRY_NAMES) -> dict[str, Any]:
    import importlib

    from nexus.core import tool_registry

    wanted = set(names)
    for mod, keys in PPT12_MODULES.items():
        if wanted & set(keys):
            importlib.import_module(f'{__name__}.{mod}')
    return {name: tool_registry.get(name) for name in names}


__all__ = [
    'CODE_WORKDIR_STATE_KEY',
    'IMAGE_STATE_KEY',
    'PPT12_ALIAS_TOOLS',
    'PPT12_ALL_REGISTRY_NAMES',
    'PPT12_MODEL_NAMES',
    'PPT12_MODULES',
    'PPT12_PRESET',
    'PPT12_TOOLS',
    'expand_agent_tools',
    'load_ppt12',
]
