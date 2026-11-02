"""LibreOffice Impress getters."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.file import get_vm_file


@getter_registry.register()
async def get_background_image_in_slide(runtime: Any, config: dict[str, Any]) -> str | None:
    return await _extract_slide_relationship(runtime, config, relation_kind='image')


@getter_registry.register()
async def get_audio_in_slide(runtime: Any, config: dict[str, Any]) -> str | None:
    return await _extract_slide_relationship(runtime, config, relation_kind='audio')


async def _extract_slide_relationship(runtime: Any, config: dict[str, Any], *, relation_kind: str) -> str | None:
    ppt_path = str(config['ppt_file_path'])
    slide_index = int(config['slide_index'])
    local_ppt = await get_vm_file(runtime, {**config, 'path': ppt_path, 'dest': os.path.basename(ppt_path)})
    if not local_ppt:
        return None

    slide_rels = f'ppt/slides/_rels/slide{slide_index + 1}.xml.rels'
    with zipfile.ZipFile(str(local_ppt), 'r') as archive:
        if slide_rels not in archive.namelist():
            return None
        with archive.open(slide_rels) as handle:
            root = ET.parse(handle).getroot()
        namespaces = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
        for rel in root.findall('r:Relationship', namespaces):
            if relation_kind not in rel.attrib.get('Type', ''):
                continue
            target = rel.attrib.get('Target', '')
            if target.startswith('..'):
                archive_path = os.path.normpath(os.path.join('ppt/slides', target)).replace('\\', '/')
                archive.extract(archive_path, os.path.dirname(str(local_ppt)))
                return os.path.join(os.path.dirname(str(local_ppt)), archive_path)
            if target.startswith('file://'):
                result = await get_vm_file(runtime, {**config, 'path': target[7:], 'dest': config['dest']})
                return str(result) if result else None
    return None
