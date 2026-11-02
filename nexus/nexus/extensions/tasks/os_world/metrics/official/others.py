# ruff: noqa
import logging
import os
import os.path
import re
import zipfile
from typing import List, Dict
from typing import Union, TypeVar

import lxml.html
from lxml.html import HtmlElement
from mutagen.easyid3 import EasyID3

from .general import diff_text_file
from .utils import _match_value_to_rule

logger = logging.getLogger('desktopenv.metric.others')


def process_epub(filename: str) -> List[str]:
    file_list: List[str] = []

    base_dir: str = filename + '.dir'
    os.makedirs(base_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(filename, 'r') as z_f:
            # Get list of all files in the zip archive
            zip_file_list = z_f.namelist()

            # Process toc.ncx if it exists
            if 'toc.ncx' in zip_file_list:
                with z_f.open('toc.ncx') as in_f, open(os.path.join(base_dir, 'toc.ncx'), 'w') as out_f:
                    contents: str = in_f.read().decode()
                    contents = contents.splitlines()
                    for l in contents:
                        if 'navPoint' not in l:
                            out_f.write(l + '\n')
                file_list.append(os.path.join(base_dir, 'toc.ncx'))
            else:
                logger.debug('toc.ncx not found in epub file: %s', filename)

            # Process content.opf if it exists
            if 'content.opf' in zip_file_list:
                with z_f.open('content.opf') as in_f, open(os.path.join(base_dir, 'content.opf'), 'w') as out_f:
                    contents: str = in_f.read().decode()
                    contents = contents.splitlines()
                    for l in contents:
                        if 'dc:identifier' not in l:
                            out_f.write(l + '\n')
                file_list.append(os.path.join(base_dir, 'content.opf'))
            else:
                logger.debug('content.opf not found in epub file: %s', filename)
            for f_n in z_f.namelist():
                if f_n.endswith('.html'):
                    with z_f.open(f_n) as in_f, open(os.path.join(base_dir, f_n), 'w') as out_f:
                        html: HtmlElement = lxml.html.fromstring(
                            ''.join(filter(lambda ch: ch != '\n' and ch != '\r', in_f.read().decode())).encode()
                        )
                        out_f.write(lxml.html.tostring(html, pretty_print=True, encoding='unicode'))
                    file_list.append(os.path.join(base_dir, f_n))
        logger.debug('%s: %s', filename, file_list)
        return list(sorted(file_list))
    except zipfile.BadZipFile:
        return []


def compare_epub(result: str, expected: str) -> float:
    if result is None:
        return 0.0
    result_files: List[str] = process_epub(result)
    expected_files: List[str] = process_epub(expected)

    metric: float = 0.0
    for f1, f2 in zip(result_files, expected_files):
        current_metric: float = diff_text_file(f1, f2)
        logger.debug('%s vs %s: %f', f1, f2, current_metric)
        metric += current_metric
    if len(result_files) > 0:
        metric /= len(result_files)
    return metric


V = TypeVar('Value')


def check_mp3_meta(result: str, meta: Dict[str, Dict[str, Union[str, V]]]) -> bool:
    # checks using _match_value_to_rule
    if result is None:
        return 0.0

    try:
        id3_dict = EasyID3(result)
    except Exception:
        logger.warning(f'Failed to read ID3 tags from {result}, returning 0')
        return 0.0
    metric: bool = True
    for k, r in meta.items():
        value = id3_dict.get(k, '')
        if isinstance(value, list):
            value: str = ','.join(value)
        logger.debug('%s.%s: %s', result, k, value)
        metric = metric and _match_value_to_rule(value, r)
    return float(metric)


def _norm_es(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip().lower()


def _norm_key_es(s: str) -> str:
    return re.sub(r'[^0-9a-z]+', '', s.lower())


def _parse_epub_es(path: str):
    import posixpath as _pp_es
    import zipfile as _zf_es

    import lxml.etree as _lxml_et_es
    import lxml.html as _lxml_html_es

    if path is None:
        return None
    try:
        zf = _zf_es.ZipFile(path, 'r')
    except Exception:
        return None
    with zf:
        names = zf.namelist()

        def read(name):
            try:
                return zf.read(name)
            except KeyError:
                return None

        opf_name = None
        raw = read('META-INF/container.xml')
        if raw:
            try:
                tree = _lxml_et_es.fromstring(raw)
                rootfiles = tree.xpath("//*[local-name()='rootfile']/@full-path")
                if rootfiles:
                    opf_name = rootfiles[0]
            except _lxml_et_es.XMLSyntaxError:
                pass
        if not opf_name or opf_name not in names:
            candidates = [name for name in names if name.lower().endswith('.opf')]
            opf_name = candidates[0] if candidates else None
        if not opf_name:
            return None

        opf_raw = read(opf_name)
        if opf_raw is None:
            return None
        try:
            opf = _lxml_et_es.fromstring(opf_raw)
        except _lxml_et_es.XMLSyntaxError:
            return None
        opf_dir = _pp_es.dirname(opf_name)

        def resolve(href):
            return _pp_es.normpath(_pp_es.join(opf_dir, href)) if opf_dir else _pp_es.normpath(href)

        titles = opf.xpath("//*[local-name()='metadata']/*[local-name()='title']/text()")
        title = titles[0].strip() if titles else ''

        manifest = {}
        for item in opf.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
            manifest[item.get('id', '')] = {
                'href': item.get('href', ''),
                'media': item.get('media-type', ''),
                'props': item.get('properties', '') or '',
            }
        spine_hrefs = []
        for itemref in opf.xpath("//*[local-name()='spine']/*[local-name()='itemref']"):
            item = manifest.get(itemref.get('idref', ''))
            if item and 'nav' not in item['props']:
                spine_hrefs.append(resolve(item['href']))

        def doc_text(name):
            raw_doc = read(name)
            if not raw_doc:
                return ''
            try:
                root = _lxml_html_es.fromstring(raw_doc)
                return _norm_es(root.text_content())
            except Exception:
                return ''

        docs = [{'href': href, 'text': doc_text(href)} for href in spine_hrefs]

        toc = []
        ncx_names = [resolve(item['href']) for item in manifest.values() if item['media'] == 'application/x-dtbncx+xml']
        ncx_names += [name for name in names if name.lower().endswith('.ncx') and name not in ncx_names]
        for name in ncx_names:
            raw_ncx = read(name)
            if not raw_ncx:
                continue
            try:
                ncx = _lxml_et_es.fromstring(raw_ncx)
                toc = [
                    _norm_es(text) for text in ncx.xpath("//*[local-name()='navPoint']//*[local-name()='text']/text()")
                ]
            except _lxml_et_es.XMLSyntaxError:
                pass
            if toc:
                break
        if not toc:
            for item in manifest.values():
                if 'nav' in item['props']:
                    raw_nav = read(resolve(item['href']))
                    if raw_nav:
                        try:
                            nav = _lxml_html_es.fromstring(raw_nav)
                            toc = [_norm_es(link.text_content()) for link in nav.xpath('//nav//a | //a')]
                        except Exception:
                            pass
                    break
    return {'title': title, 'docs': docs, 'toc': toc}


def compare_epub_semantic(result: str, expected: str, **options) -> float:
    from rapidfuzz import fuzz as _fuzz_es

    raw = bool(options.get('raw_score', False))

    if result is None:
        return 0.0
    got = _parse_epub_es(result)
    gold = _parse_epub_es(expected)
    if gold is None:
        logger.error('compare_epub_semantic: gold unparseable: %s', expected)
        return 0.0
    if got is None:
        logger.info('compare_epub_semantic: result unparseable: %s', result)
        return 0.0

    gold_chaps = [doc['text'] for doc in gold['docs'] if len(doc['text']) >= 50]
    res_docs = [doc['text'] for doc in got['docs']]
    if not gold_chaps:
        logger.error('compare_epub_semantic: gold has no chapter text >= 50 chars')
        return 0.0

    s_struct = 1.0 if res_docs else 0.0

    if got['title'] and gold['title']:
        s_title = _fuzz_es.ratio(_norm_key_es(got['title']), _norm_key_es(gold['title'])) / 100.0
    else:
        s_title = 1.0 if not got['title'] else 0.0

    used = set()
    sims = []
    match_idx = []
    for gold_chapter in gold_chaps:
        best_i, best_s = -1, 0.0
        for idx, text in enumerate(res_docs):
            if idx in used or not text:
                continue
            score = _fuzz_es.ratio(gold_chapter, text) / 100.0
            if score > best_s:
                best_i, best_s = idx, score
        sims.append(best_s)
        if best_i >= 0:
            used.add(best_i)
            match_idx.append(best_i)
    s_body = sum(sims) / len(sims) if sims else 0.0
    s_order = 1.0 if len(match_idx) == len(gold_chaps) and match_idx == sorted(match_idx) else 0.0

    gates_pass = all(score >= 0.5 for score in sims) and s_body >= 0.6
    if not gates_pass:
        gold_concat = ' '.join(gold_chaps)
        res_concat = ' '.join(text for text in res_docs if text)
        concat_sim = _fuzz_es.ratio(gold_concat, res_concat) / 100.0
        if concat_sim >= 0.95:
            s_body = concat_sim
            s_order = 1.0
            sims = [concat_sim]
            gates_pass = True

    s_toc = min(len([text for text in got['toc'] if text]) / len(gold_chaps), 1.0)
    score = 0.05 * s_struct + 0.10 * s_title + 0.20 * s_order + 0.60 * s_body + 0.05 * s_toc

    logger.info(
        'compare_epub_semantic %s: struct=%.2f title=%.2f order=%.2f body=%.3f toc=%.2f -> %.3f gates_pass=%s',
        result,
        s_struct,
        s_title,
        s_order,
        s_body,
        s_toc,
        score,
        gates_pass,
    )

    if raw:
        return score
    if not gates_pass and not (all(score >= 0.5 for score in sims) and s_body >= 0.6):
        return 0.0
    return 1.0 if score >= 0.85 else 0.0
