#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import os

from pathlib import Path
from typing import Any

from aiohttp import web
from toolathlon_mock_utils import mutate_store, slugify, utc_now_iso


STORE_PATH_ENV = 'NEXUS_TOOLATHLON_WOOCOMMERCE_STORE_PATH'
MEDIA_DIR_ENV = 'NEXUS_TOOLATHLON_WOOCOMMERCE_MEDIA_DIR'
DEFAULT_PORT = 10003
DEFAULT_CUSTOMER_ID = 1
DEFAULT_CATEGORY_ID = 15
DEFAULT_REST_NONCE = 'mock-rest-nonce'
DEFAULT_POST_ID = 1


def _store_path() -> str:
    value = os.getenv(STORE_PATH_ENV)
    if not value:
        raise RuntimeError(f'{STORE_PATH_ENV} is not set')
    return value


def _media_dir() -> Path:
    value = os.getenv(MEDIA_DIR_ENV)
    if value:
        return Path(value)
    return Path('/tmp/nexus-toolathlon-woocommerce-media')


def _default_state() -> dict[str, Any]:
    return {
        'version': 1,
        'next_ids': {
            'product': 1,
            'customer': 2,
            'order': 1,
            'category': DEFAULT_CATEGORY_ID + 1,
            'attribute': 1,
            'term': 1,
            'variation': 1,
            'coupon': 1,
            'media': 1,
            'note': 1,
            'refund': 1,
            'review': 1,
            'tag': 1,
            'post': DEFAULT_POST_ID + 1,
            'shipping_zone': 1,
            'shipping_method': 1,
            'tax': 1,
        },
        'stores': {},
    }


def _next_id_default(key: str) -> int:
    if key == 'post':
        return DEFAULT_POST_ID + 1
    return 1


def _next_id(state: dict[str, Any], key: str) -> int:
    next_ids = state.setdefault('next_ids', {})
    value = int(next_ids.get(key, _next_id_default(key)))
    next_ids[key] = value + 1
    return value


def _json(data: Any, *, status: int = 200) -> web.Response:
    return web.json_response(data, status=status)


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _compact_store_key(raw: str | None) -> str:
    text = (raw or '').strip().strip('/')
    return text or 'default'


def _default_customer_payload() -> dict[str, Any]:
    return {
        'id': DEFAULT_CUSTOMER_ID,
        'email': 'customer1@woocommerce.mock',
        'first_name': 'Default',
        'last_name': 'Customer',
        'role': 'customer',
        'username': 'customer1',
        'date_created': utc_now_iso(),
        'billing': {
            'email': 'customer1@woocommerce.mock',
            'first_name': 'Default',
            'last_name': 'Customer',
        },
        'shipping': {},
        'meta_data': [],
    }


def _default_shipping_zone_payload() -> dict[str, Any]:
    return {
        'id': 0,
        'name': 'Locations not covered by your other zones',
        'order': 0,
        'locations': [],
        'methods': {},
    }


def _default_post_record() -> dict[str, Any]:
    now = utc_now_iso()
    return {
        'id': DEFAULT_POST_ID,
        'date': now,
        'date_gmt': now,
        'modified': now,
        'modified_gmt': now,
        'slug': 'hello-world',
        'status': 'publish',
        'title': 'Hello World',
        'content': 'Welcome to WordPress. This is your first post. Edit or delete it, then start writing!',
        'excerpt': '',
        'author': 1,
        'featured_media': 0,
        'comment_status': 'open',
        'ping_status': 'open',
        'sticky': False,
        'template': '',
        'format': 'standard',
        'meta': [],
        'categories': [],
        'tags': [],
    }


def _store_key(request: web.Request) -> str:
    return _compact_store_key(request.match_info.get('store'))


def _site_root(request: web.Request, store_key: str) -> str:
    base = f'{request.scheme}://{request.host}'
    return base if store_key == 'default' else f'{base}/{store_key}'


def _ensure_store(state: dict[str, Any], store_key: str) -> dict[str, Any]:
    stores = state.setdefault('stores', {})
    store = stores.get(store_key)
    if store is None:
        store = {
            'products': {},
            'customers': {str(DEFAULT_CUSTOMER_ID): _default_customer_payload()},
            'orders': {},
            'categories': {
                str(DEFAULT_CATEGORY_ID): {
                    'id': DEFAULT_CATEGORY_ID,
                    'name': 'Uncategorized',
                    'slug': 'uncategorized',
                    'description': '',
                    'parent': 0,
                    'count': 0,
                }
            },
            'attributes': {},
            'tags': {},
            'coupons': {},
            'media': {},
            'reviews': {},
            'sessions': {},
            'posts': {str(DEFAULT_POST_ID): _default_post_record()},
            'shipping_zones': {'0': _default_shipping_zone_payload()},
            'taxes': {},
        }
        stores[store_key] = store
    store.setdefault('products', {})
    store.setdefault('customers', {})
    _ensure_default_customer(store)
    store.setdefault(
        'categories',
        {
            str(DEFAULT_CATEGORY_ID): {
                'id': DEFAULT_CATEGORY_ID,
                'name': 'Uncategorized',
                'slug': 'uncategorized',
                'description': '',
                'parent': 0,
                'count': 0,
            }
        },
    )
    store.setdefault('attributes', {})
    store.setdefault('tags', {})
    store.setdefault('coupons', {})
    store.setdefault('media', {})
    store.setdefault('reviews', {})
    store.setdefault('sessions', {})
    posts = store.setdefault('posts', {})
    posts.setdefault(str(DEFAULT_POST_ID), _default_post_record())
    shipping_zones = store.setdefault('shipping_zones', {})
    shipping_zones.setdefault('0', _default_shipping_zone_payload())
    store.setdefault('taxes', {})
    return store


def _ensure_default_customer(store: dict[str, Any]) -> dict[str, Any]:
    customer = store['customers'].get(str(DEFAULT_CUSTOMER_ID))
    if customer is None:
        customer = _default_customer_payload()
        store['customers'][str(DEFAULT_CUSTOMER_ID)] = customer
    return customer


def _media_file_path(store_key: str, media_id: int, filename: str) -> Path:
    safe_name = slugify(filename or 'file')
    store_dir = _media_dir() / store_key
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / f'{media_id}-{safe_name}'


def _basic_auth_user(request: web.Request) -> str | None:
    header = request.headers.get('Authorization', '')
    if not header.startswith('Basic '):
        return None
    try:
        decoded = base64.b64decode(header[len('Basic ') :].strip()).decode('utf-8')
    except Exception:
        return None
    if ':' not in decoded:
        return decoded or None
    username, _password = decoded.split(':', 1)
    return username or None


def _product_sales(store: dict[str, Any]) -> tuple[dict[int, int], dict[int, int]]:
    product_totals: dict[int, int] = {}
    variation_totals: dict[int, int] = {}
    for order in store['orders'].values():
        if str(order.get('status') or '').lower() in {'cancelled', 'failed', 'trash'}:
            continue
        for item in order.get('line_items', []):
            quantity = max(0, _coerce_int(item.get('quantity'), 0))
            product_id = _coerce_int(item.get('product_id'))
            variation_id = _coerce_int(item.get('variation_id'))
            if product_id:
                product_totals[product_id] = product_totals.get(product_id, 0) + quantity
            if variation_id:
                variation_totals[variation_id] = variation_totals.get(variation_id, 0) + quantity
    return product_totals, variation_totals


def _paginate(items: list[Any], request: web.Request, *, default_per_page: int = 10) -> list[Any]:
    page = max(1, _coerce_int(request.query.get('page'), 1))
    per_page = max(1, min(100, _coerce_int(request.query.get('per_page'), default_per_page)))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end]


def _match_search(haystack: list[str], needle: str) -> bool:
    if not needle:
        return True
    value = needle.lower()
    return any(value in (item or '').lower() for item in haystack)


def _merge_meta_data(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in existing:
        key = str(item.get('key') or '')
        if key:
            merged[key] = dict(item)
    for item in updates:
        key = str(item.get('key') or '')
        if key:
            merged[key] = dict(item)
    return list(merged.values())


def _find_customer_by_email(store: dict[str, Any], email: str) -> dict[str, Any] | None:
    for customer in store['customers'].values():
        if str(customer.get('email') or '').lower() == email.lower():
            return customer
    return None


def _resolve_customer(store: dict[str, Any], state: dict[str, Any], order_data: dict[str, Any]) -> dict[str, Any]:
    customer_id = _coerce_int(order_data.get('customer_id'))
    if customer_id and str(customer_id) in store['customers']:
        return store['customers'][str(customer_id)]

    billing = order_data.get('billing', {}) or {}
    email = str(billing.get('email') or '').strip()
    if email:
        existing = _find_customer_by_email(store, email)
        if existing is not None:
            return existing
        customer_id = _next_id(state, 'customer')
        customer = {
            'id': customer_id,
            'email': email,
            'first_name': billing.get('first_name', ''),
            'last_name': billing.get('last_name', ''),
            'role': 'customer',
            'username': email.split('@', 1)[0] or f'customer{customer_id}',
            'date_created': utc_now_iso(),
            'billing': dict(billing),
            'shipping': dict(order_data.get('shipping') or {}),
            'meta_data': [],
        }
        store['customers'][str(customer_id)] = customer
        return customer

    return _ensure_default_customer(store)


def _resolve_category_refs(state: dict[str, Any], store: dict[str, Any], refs: Any) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        return []
    resolved: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ref in refs:
        category: dict[str, Any] | None = None
        if isinstance(ref, dict):
            category_id = _coerce_int(ref.get('id'))
            if category_id and str(category_id) in store['categories']:
                category = store['categories'][str(category_id)]
            else:
                name = str(ref.get('name') or '').strip()
                slug = slugify(str(ref.get('slug') or name))
                if name:
                    for existing in store['categories'].values():
                        if existing.get('name') == name or existing.get('slug') == slug:
                            category = existing
                            break
                    if category is None:
                        category_id = _next_id(state, 'category')
                        category = {
                            'id': category_id,
                            'name': name,
                            'slug': slug,
                            'description': str(ref.get('description') or ''),
                            'parent': _coerce_int(ref.get('parent'), 0),
                            'count': 0,
                        }
                        store['categories'][str(category_id)] = category
        elif isinstance(ref, int):
            category = store['categories'].get(str(ref))
        if category is None:
            continue
        category_id = _coerce_int(category.get('id'))
        if category_id and category_id not in seen:
            seen.add(category_id)
            resolved.append({'id': category_id, 'name': category.get('name', ''), 'slug': category.get('slug', '')})
    return resolved


def _resolve_images(store: dict[str, Any], refs: Any) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        return []
    images: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ref in refs:
        media: dict[str, Any] | None = None
        if isinstance(ref, dict):
            media_id = _coerce_int(ref.get('id'))
            if media_id:
                media = store['media'].get(str(media_id))
            elif ref.get('src'):
                src = str(ref.get('src'))
                for candidate in store['media'].values():
                    if candidate.get('source_url') == src:
                        media = candidate
                        break
        if media is None:
            continue
        media_id = _coerce_int(media.get('id'))
        if media_id and media_id not in seen:
            seen.add(media_id)
            images.append({
                'id': media_id,
                'src': media.get('source_url', ''),
                'name': media.get('filename', ''),
                'alt': media.get('alt_text', ''),
                'position': len(images),
            })
    return images


def _category_payload(category: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': category['id'],
        'name': category.get('name', ''),
        'slug': category.get('slug', ''),
        'description': category.get('description', ''),
        'parent': category.get('parent', 0),
        'count': category.get('count', 0),
    }


def _attribute_payload(attribute: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': attribute['id'],
        'name': attribute.get('name', ''),
        'slug': attribute.get('slug', ''),
        'type': attribute.get('type', 'select'),
        'order_by': attribute.get('order_by', 'menu_order'),
        'has_archives': bool(attribute.get('has_archives', False)),
    }


def _term_payload(term: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': term['id'],
        'name': term.get('name', ''),
        'slug': term.get('slug', ''),
        'description': term.get('description', ''),
        'menu_order': term.get('menu_order', 0),
        'count': term.get('count', 0),
    }


def _media_payload(request: web.Request, store_key: str, media: dict[str, Any]) -> dict[str, Any]:
    source_url = (
        media.get('source_url')
        or f'{_site_root(request, store_key)}/mock-media/{media["id"]}/{media.get("filename", "media.bin")}'
    )
    return {
        'id': media['id'],
        'date': media.get('date', utc_now_iso()),
        'slug': media.get('slug', slugify(media.get('filename', 'media'))),
        'type': 'attachment',
        'link': source_url,
        'title': {'rendered': media.get('title', media.get('filename', ''))},
        'media_type': media.get('media_type', 'image'),
        'mime_type': media.get('mime_type', 'application/octet-stream'),
        'source_url': source_url,
        'alt_text': media.get('alt_text', ''),
    }


def _variation_payload(store: dict[str, Any], variation: dict[str, Any]) -> dict[str, Any]:
    image = variation.get('image') or {}
    images = (
        [
            {
                'id': image.get('id'),
                'src': image.get('src', ''),
                'name': image.get('name', ''),
                'alt': image.get('alt', ''),
                'position': 0,
            }
        ]
        if image
        else []
    )
    return {
        'id': variation['id'],
        'product_id': variation['product_id'],
        'regular_price': str(variation.get('regular_price', '')),
        'price': str(variation.get('price', variation.get('regular_price', ''))),
        'stock_quantity': variation.get('stock_quantity'),
        'manage_stock': bool(variation.get('manage_stock', False)),
        'stock_status': variation.get('stock_status', 'instock'),
        'attributes': list(variation.get('attributes', [])),
        'image': dict(image),
        'images': images,
        'meta_data': list(variation.get('meta_data', [])),
        'date_created': variation.get('date_created', utc_now_iso()),
    }


def _product_payload(store: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
    product_totals, _variation_totals = _product_sales(store)
    categories = list(product.get('categories', []))
    images = list(product.get('images', []))
    return {
        'id': product['id'],
        'name': product.get('name', ''),
        'slug': product.get('slug', slugify(product.get('name', 'product'))),
        'type': product.get('type', 'simple'),
        'status': product.get('status', 'publish'),
        'featured': bool(product.get('featured', False)),
        'catalog_visibility': product.get('catalog_visibility', 'visible'),
        'description': product.get('description', ''),
        'short_description': product.get('short_description', ''),
        'sku': product.get('sku', ''),
        'price': str(product.get('price', product.get('regular_price', ''))),
        'regular_price': str(product.get('regular_price', product.get('price', ''))),
        'sale_price': str(product.get('sale_price', '')),
        'on_sale': bool(product.get('on_sale', False)),
        'manage_stock': bool(product.get('manage_stock', False)),
        'stock_quantity': product.get('stock_quantity'),
        'stock_status': product.get('stock_status', 'instock'),
        'categories': categories,
        'images': images,
        'attributes': list(product.get('attributes', [])),
        'meta_data': list(product.get('meta_data', [])),
        'date_created': product.get('date_created', utc_now_iso()),
        'total_sales': product_totals.get(int(product['id']), 0),
        'variations': [variation['id'] for variation in product.get('variations', {}).values()],
    }


def _customer_payload(customer: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': customer['id'],
        'email': customer.get('email', ''),
        'first_name': customer.get('first_name', ''),
        'last_name': customer.get('last_name', ''),
        'role': customer.get('role', 'customer'),
        'username': customer.get('username', ''),
        'date_created': customer.get('date_created', utc_now_iso()),
        'billing': dict(customer.get('billing', {})),
        'shipping': dict(customer.get('shipping', {})),
        'meta_data': list(customer.get('meta_data', [])),
    }


def _order_line_items(store: dict[str, Any], line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in line_items:
        product_id = _coerce_int(item.get('product_id'))
        variation_id = _coerce_int(item.get('variation_id'))
        product = store['products'].get(str(product_id))
        variation = product.get('variations', {}).get(str(variation_id)) if product and variation_id else None
        quantity = max(1, _coerce_int(item.get('quantity'), 1))
        price = str(item.get('price') or item.get('subtotal') or (variation or product or {}).get('price') or '')
        items.append({
            'id': item.get('id') or 0,
            'product_id': product_id,
            'variation_id': variation_id,
            'quantity': quantity,
            'name': item.get('name') or (variation and product.get('name')) or (product and product.get('name')) or '',
            'sku': item.get('sku') or (variation and variation.get('sku')) or (product and product.get('sku')) or '',
            'price': price,
            'subtotal': item.get('subtotal') or price,
            'total': item.get('total') or price,
        })
    return items


def _order_payload(store: dict[str, Any], order: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': order['id'],
        'status': order.get('status', 'pending'),
        'customer_id': order.get('customer_id', DEFAULT_CUSTOMER_ID),
        'payment_method': order.get('payment_method', ''),
        'payment_method_title': order.get('payment_method_title', ''),
        'billing': dict(order.get('billing', {})),
        'shipping': dict(order.get('shipping', {})),
        'line_items': _order_line_items(store, list(order.get('line_items', []))),
        'meta_data': list(order.get('meta_data', [])),
        'date_created': order.get('date_created', utc_now_iso()),
        'date_modified': order.get('date_modified', order.get('date_created', utc_now_iso())),
        'notes': list(order.get('notes', [])),
        'refunds': list(order.get('refunds', [])),
    }


def _coupon_payload(coupon: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': coupon['id'],
        'code': coupon.get('code', ''),
        'description': coupon.get('description', ''),
        'amount': coupon.get('amount', '0'),
        'discount_type': coupon.get('discount_type', 'fixed_cart'),
        'meta_data': list(coupon.get('meta_data', [])),
        'date_created': coupon.get('date_created', utc_now_iso()),
    }


def _rendered_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get('rendered') or value.get('raw') or '')
    return str(value or '')


def _shipping_zone_payload(zone: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': zone['id'],
        'name': zone.get('name', ''),
        'order': _coerce_int(zone.get('order'), 0),
        'locations': list(zone.get('locations', [])),
        '_links': {},
    }


def _shipping_zone_method_payload(method: dict[str, Any]) -> dict[str, Any]:
    method_id = _coerce_int(method.get('id') or method.get('instance_id'))
    raw_method_id = str(method.get('method_id') or 'flat_rate')
    return {
        'id': method_id,
        'instance_id': method_id,
        'title': method.get('title', raw_method_id.replace('_', ' ').title()),
        'order': _coerce_int(method.get('order'), 0),
        'enabled': bool(method.get('enabled', True)),
        'method_id': raw_method_id,
        'method_title': method.get('method_title', raw_method_id.replace('_', ' ').title()),
        'settings': dict(method.get('settings', {})),
        '_links': {},
    }


def _post_payload(request: web.Request, store_key: str, post: dict[str, Any]) -> dict[str, Any]:
    post_id = _coerce_int(post.get('id'))
    title = _rendered_text(post.get('title'))
    content = _rendered_text(post.get('content'))
    excerpt = _rendered_text(post.get('excerpt'))
    link = post.get('link') or f'{_site_root(request, store_key)}/?p={post_id}'
    guid = post.get('guid') or link
    return {
        'id': post_id,
        'date': post.get('date', utc_now_iso()),
        'date_gmt': post.get('date_gmt', post.get('date', utc_now_iso())),
        'guid': {'rendered': guid},
        'modified': post.get('modified', post.get('date', utc_now_iso())),
        'modified_gmt': post.get('modified_gmt', post.get('modified', post.get('date', utc_now_iso()))),
        'slug': post.get('slug', slugify(title or f'post-{post_id}')),
        'status': post.get('status', 'publish'),
        'type': 'post',
        'link': link,
        'title': {'rendered': title},
        'content': {'rendered': content, 'protected': False},
        'excerpt': {'rendered': excerpt, 'protected': False},
        'author': _coerce_int(post.get('author'), 1),
        'featured_media': _coerce_int(post.get('featured_media'), 0),
        'comment_status': post.get('comment_status', 'open'),
        'ping_status': post.get('ping_status', 'open'),
        'sticky': bool(post.get('sticky', False)),
        'template': post.get('template', ''),
        'format': post.get('format', 'standard'),
        'meta': list(post.get('meta', [])),
        'categories': list(post.get('categories', [])),
        'tags': list(post.get('tags', [])),
        '_links': {},
    }


def _tax_payload(tax: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': tax['id'],
        'country': tax.get('country', ''),
        'state': tax.get('state', ''),
        'postcode': tax.get('postcode', ''),
        'city': tax.get('city', ''),
        'rate': str(tax.get('rate', '0')),
        'name': tax.get('name', ''),
        'priority': _coerce_int(tax.get('priority'), 1),
        'compound': bool(tax.get('compound', False)),
        'shipping': bool(tax.get('shipping', True)),
        'order': _coerce_int(tax.get('order'), 0),
        'class': tax.get('class', ''),
        '_links': {},
    }


async def _request_json(request: web.Request) -> dict[str, Any]:
    if request.content_type.startswith('application/json'):
        try:
            payload = await request.json()
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    try:
        form = await request.post()
        return dict(form)
    except Exception:
        return {}


async def health(_: web.Request) -> web.Response:
    return _json({'status': 'ok'})


async def site_root(request: web.Request) -> web.Response:
    return web.Response(text='Mock WooCommerce Site', content_type='text/plain')


async def wp_login_get(_: web.Request) -> web.Response:
    html = '<html><body><form><input type="hidden" name="_wpnonce" value="mock-login-nonce" /></form></body></html>'
    return web.Response(text=html, content_type='text/html')


async def wp_login_post(request: web.Request) -> web.Response:
    payload = await request.post()
    username = str(payload.get('log') or _basic_auth_user(request) or 'wordpress')
    response = web.Response(status=302, headers={'Location': request.query.get('redirect_to', '/')})
    response.set_cookie('wordpress_logged_in_mock', username)
    return response


async def wp_rest_nonce(_: web.Request) -> web.Response:
    return web.Response(text=DEFAULT_REST_NONCE, content_type='text/plain')


async def media_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            media_id = _next_id(state, 'media')
            disposition = request.headers.get('Content-Disposition', '')
            filename = 'image.jpg'
            if 'filename=' in disposition:
                filename = disposition.split('filename=', 1)[1].strip().strip('"')
            content = request._read_bytes if hasattr(request, '_read_bytes') else None
            if content is None:
                raise RuntimeError('request body not read')
            path = _media_file_path(store_key, media_id, filename)
            path.write_bytes(content)
            media = {
                'id': media_id,
                'filename': filename,
                'title': Path(filename).stem,
                'slug': slugify(Path(filename).stem),
                'mime_type': request.headers.get('Content-Type', 'application/octet-stream'),
                'media_type': 'image',
                'alt_text': '',
                'file_path': str(path),
                'date': utc_now_iso(),
                'source_url': f'{_site_root(request, store_key)}/mock-media/{media_id}/{filename}',
            }
            store['media'][str(media_id)] = media
            return 201, _media_payload(request, store_key, media)

        content = await request.read()
        request._read_bytes = content  # type: ignore[attr-defined]
        status, payload = mutate_store(_store_path(), _default_state, create)
        return _json(payload, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_media_payload(request, store_key, media) for media in store['media'].values()]
        if request.query.get('media_type') == 'image':
            rows = [row for row in rows if row.get('media_type') == 'image']
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def media_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    media_id = _coerce_int(request.match_info['media_id'])

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        media = store['media'].get(str(media_id))
        if media is None:
            return 404, {'message': 'media not found'}
        if request.method == 'DELETE':
            store['media'].pop(str(media_id), None)
            with contextlib.suppress(Exception):
                Path(media.get('file_path', '')).unlink(missing_ok=True)
            return 200, {'deleted': True, 'previous': _media_payload(request, store_key, media)}
        return 200, _media_payload(request, store_key, media)

    status, payload = mutate_store(_store_path(), _default_state, handle)
    return _json(payload, status=status)


async def media_file(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    media_id = _coerce_int(request.match_info['media_id'])

    def build(state: dict[str, Any]) -> tuple[int, bytes, str]:
        store = _ensure_store(state, store_key)
        media = store['media'].get(str(media_id))
        if media is None:
            return 404, b'', 'text/plain'
        path = Path(media.get('file_path', ''))
        if not path.exists():
            return 404, b'', 'text/plain'
        return 200, path.read_bytes(), media.get('mime_type', 'application/octet-stream')

    status, content, content_type = mutate_store(_store_path(), _default_state, build)
    return web.Response(status=status, body=content, content_type=content_type)


async def system_status(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    root = _site_root(request, store_key)
    payload = {
        'environment': {
            'home_url': root,
            'site_url': root,
            'version': '9.9-mock',
        },
        'settings': {'api_enabled': True},
        'user': {'username': _basic_auth_user(request) or 'mock-user'},
    }
    return _json(payload)


async def products_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            product_id = _next_id(state, 'product')
            categories = _resolve_category_refs(state, store, payload.get('categories'))
            images = _resolve_images(store, payload.get('images'))
            product = {
                'id': product_id,
                'name': payload.get('name', f'Product {product_id}'),
                'slug': slugify(str(payload.get('slug') or payload.get('name') or f'product-{product_id}')),
                'type': payload.get('type', 'simple'),
                'status': payload.get('status', 'publish'),
                'featured': bool(payload.get('featured', False)),
                'catalog_visibility': payload.get('catalog_visibility', 'visible'),
                'description': payload.get('description', ''),
                'short_description': payload.get('short_description', ''),
                'sku': payload.get('sku', ''),
                'price': str(payload.get('price') or payload.get('regular_price') or ''),
                'regular_price': str(payload.get('regular_price') or payload.get('price') or ''),
                'sale_price': str(payload.get('sale_price') or ''),
                'on_sale': bool(payload.get('on_sale', False)),
                'manage_stock': bool(payload.get('manage_stock', False)),
                'stock_quantity': payload.get('stock_quantity'),
                'stock_status': payload.get('stock_status', 'instock'),
                'categories': categories,
                'images': images,
                'attributes': list(payload.get('attributes') or []),
                'meta_data': list(payload.get('meta_data') or []),
                'date_created': payload.get('date_created', utc_now_iso()),
                'variations': {},
            }
            store['products'][str(product_id)] = product
            return 201, _product_payload(store, product)

        status, product = mutate_store(_store_path(), _default_state, create)
        return _json(product, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_product_payload(store, product) for product in store['products'].values()]
        search = str(request.query.get('search') or '').strip()
        sku = str(request.query.get('sku') or '').strip()
        status_filter = str(request.query.get('status') or '').strip()
        stock_status = str(request.query.get('stock_status') or '').strip()
        category = _coerce_int(request.query.get('category'))
        featured = request.query.get('featured')
        on_sale = request.query.get('on_sale')
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if search and not _match_search(
                [row.get('name', ''), row.get('sku', ''), row.get('description', '')], search
            ):
                continue
            if sku and str(row.get('sku') or '') != sku:
                continue
            if status_filter and str(row.get('status') or '') != status_filter:
                continue
            if stock_status and str(row.get('stock_status') or '') != stock_status:
                continue
            if category and not any(_coerce_int(cat.get('id')) == category for cat in row.get('categories', [])):
                continue
            if featured is not None and bool(row.get('featured', False)) != _boolish(featured):
                continue
            if on_sale is not None and bool(row.get('on_sale', False)) != _boolish(on_sale):
                continue
            filtered.append(row)
        filtered.sort(key=lambda row: int(row['id']))
        return _paginate(filtered, request)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def product_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    product_id = _coerce_int(request.match_info['product_id'])
    payload = await _request_json(request) if request.method in {'PUT'} else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        product = store['products'].get(str(product_id))
        if product is None:
            return 404, {'message': 'product not found'}
        if request.method == 'DELETE':
            store['products'].pop(str(product_id), None)
            return 200, {'deleted': True, 'previous': _product_payload(store, product)}
        if request.method == 'PUT':
            for key in (
                'name',
                'slug',
                'type',
                'status',
                'catalog_visibility',
                'description',
                'short_description',
                'sku',
                'price',
                'regular_price',
                'sale_price',
                'stock_quantity',
                'stock_status',
            ):
                if key in payload:
                    product[key] = payload[key]
            if 'featured' in payload:
                product['featured'] = bool(payload['featured'])
            if 'on_sale' in payload:
                product['on_sale'] = bool(payload['on_sale'])
            if 'manage_stock' in payload:
                product['manage_stock'] = bool(payload['manage_stock'])
            if 'categories' in payload:
                product['categories'] = _resolve_category_refs(state, store, payload.get('categories'))
            if 'images' in payload:
                product['images'] = _resolve_images(store, payload.get('images'))
            if 'attributes' in payload:
                product['attributes'] = list(payload.get('attributes') or [])
            if 'meta_data' in payload:
                product['meta_data'] = _merge_meta_data(
                    list(product.get('meta_data', [])), list(payload.get('meta_data') or [])
                )
        return 200, _product_payload(store, product)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def products_batch(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    payload = await _request_json(request)

    def handle(state: dict[str, Any]) -> dict[str, Any]:
        store = _ensure_store(state, store_key)
        created: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []

        for item in payload.get('create') or []:
            product_id = _next_id(state, 'product')
            product = {
                'id': product_id,
                'name': item.get('name', f'Product {product_id}'),
                'slug': slugify(str(item.get('slug') or item.get('name') or f'product-{product_id}')),
                'type': item.get('type', 'simple'),
                'status': item.get('status', 'publish'),
                'featured': bool(item.get('featured', False)),
                'catalog_visibility': item.get('catalog_visibility', 'visible'),
                'description': item.get('description', ''),
                'short_description': item.get('short_description', ''),
                'sku': item.get('sku', ''),
                'price': str(item.get('price') or item.get('regular_price') or ''),
                'regular_price': str(item.get('regular_price') or item.get('price') or ''),
                'sale_price': str(item.get('sale_price') or ''),
                'on_sale': bool(item.get('on_sale', False)),
                'manage_stock': bool(item.get('manage_stock', False)),
                'stock_quantity': item.get('stock_quantity'),
                'stock_status': item.get('stock_status', 'instock'),
                'categories': _resolve_category_refs(state, store, item.get('categories')),
                'images': _resolve_images(store, item.get('images')),
                'attributes': list(item.get('attributes') or []),
                'meta_data': list(item.get('meta_data') or []),
                'date_created': item.get('date_created', utc_now_iso()),
                'variations': {},
            }
            store['products'][str(product_id)] = product
            created.append(_product_payload(store, product))

        for item in payload.get('update') or []:
            product_id = _coerce_int(item.get('id'))
            product = store['products'].get(str(product_id))
            if product is None:
                continue
            for key in (
                'name',
                'slug',
                'type',
                'status',
                'catalog_visibility',
                'description',
                'short_description',
                'sku',
                'price',
                'regular_price',
                'sale_price',
                'stock_quantity',
                'stock_status',
            ):
                if key in item:
                    product[key] = item[key]
            if 'featured' in item:
                product['featured'] = bool(item['featured'])
            if 'on_sale' in item:
                product['on_sale'] = bool(item['on_sale'])
            if 'manage_stock' in item:
                product['manage_stock'] = bool(item['manage_stock'])
            if 'categories' in item:
                product['categories'] = _resolve_category_refs(state, store, item.get('categories'))
            if 'images' in item:
                product['images'] = _resolve_images(store, item.get('images'))
            if 'attributes' in item:
                product['attributes'] = list(item.get('attributes') or [])
            if 'meta_data' in item:
                product['meta_data'] = _merge_meta_data(
                    list(product.get('meta_data', [])), list(item.get('meta_data') or [])
                )
            updated.append(_product_payload(store, product))

        for item in payload.get('delete') or []:
            product_id = _coerce_int(item)
            product = store['products'].pop(str(product_id), None)
            if product is not None:
                deleted.append({'id': product_id})

        return {'create': created, 'update': updated, 'delete': deleted}

    return _json(mutate_store(_store_path(), _default_state, handle))


async def product_categories_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            name = str(payload.get('name') or '').strip() or f'Category {_next_id(state, "category")}'
            category_id = _next_id(state, 'category')
            category = {
                'id': category_id,
                'name': name,
                'slug': slugify(str(payload.get('slug') or name)),
                'description': str(payload.get('description') or ''),
                'parent': _coerce_int(payload.get('parent'), 0),
                'count': 0,
            }
            store['categories'][str(category_id)] = category
            return 201, _category_payload(category)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_category_payload(category) for category in store['categories'].values()]
        if _boolish(request.query.get('hide_empty')):
            rows = [row for row in rows if _coerce_int(row.get('count')) > 0]
        search = str(request.query.get('search') or '').strip()
        if search:
            rows = [
                row
                for row in rows
                if _match_search([row.get('name', ''), row.get('slug', ''), row.get('description', '')], search)
            ]
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def product_category_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    category_id = _coerce_int(request.match_info['category_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        category = store['categories'].get(str(category_id))
        if category is None:
            return 404, {'message': 'category not found'}
        if request.method == 'DELETE':
            if category_id == DEFAULT_CATEGORY_ID:
                return 400, {'message': 'default category cannot be deleted'}
            store['categories'].pop(str(category_id), None)
            for product in store['products'].values():
                product['categories'] = [
                    ref for ref in product.get('categories', []) if _coerce_int(ref.get('id')) != category_id
                ]
            return 200, {'deleted': True, 'previous': _category_payload(category)}
        if request.method == 'PUT':
            for key in ('name', 'slug', 'description'):
                if key in payload:
                    category[key] = payload[key]
            if 'parent' in payload:
                category['parent'] = _coerce_int(payload.get('parent'), 0)
        return 200, _category_payload(category)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def product_categories_batch(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    payload = await _request_json(request)

    def handle(state: dict[str, Any]) -> dict[str, Any]:
        store = _ensure_store(state, store_key)
        deleted: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        created: list[dict[str, Any]] = []
        for item in payload.get('create') or []:
            category_id = _next_id(state, 'category')
            name = str(item.get('name') or f'Category {category_id}')
            category = {
                'id': category_id,
                'name': name,
                'slug': slugify(str(item.get('slug') or name)),
                'description': str(item.get('description') or ''),
                'parent': _coerce_int(item.get('parent'), 0),
                'count': 0,
            }
            store['categories'][str(category_id)] = category
            created.append(_category_payload(category))
        for item in payload.get('update') or []:
            category_id = _coerce_int(item.get('id'))
            category = store['categories'].get(str(category_id))
            if category is None:
                continue
            for key in ('name', 'slug', 'description'):
                if key in item:
                    category[key] = item[key]
            if 'parent' in item:
                category['parent'] = _coerce_int(item.get('parent'), 0)
            updated.append(_category_payload(category))
        for item in payload.get('delete') or []:
            category_id = _coerce_int(item)
            if not category_id or category_id == DEFAULT_CATEGORY_ID:
                continue
            if store['categories'].pop(str(category_id), None) is not None:
                for product in store['products'].values():
                    product['categories'] = [
                        ref for ref in product.get('categories', []) if _coerce_int(ref.get('id')) != category_id
                    ]
                deleted.append({'id': category_id})
        return {'create': created, 'update': updated, 'delete': deleted}

    return _json(mutate_store(_store_path(), _default_state, handle))


async def product_tags_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            tag_id = _next_id(state, 'tag')
            name = str(payload.get('name') or f'Tag {tag_id}')
            tag = {
                'id': tag_id,
                'name': name,
                'slug': slugify(str(payload.get('slug') or name)),
                'description': str(payload.get('description') or ''),
                'count': 0,
            }
            store['tags'][str(tag_id)] = tag
            return 201, dict(tag)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [dict(tag) for tag in store['tags'].values()]
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def product_attributes_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            attribute_id = _next_id(state, 'attribute')
            name = str(payload.get('name') or f'Attribute {attribute_id}')
            attribute = {
                'id': attribute_id,
                'name': name,
                'slug': slugify(str(payload.get('slug') or name)),
                'type': payload.get('type', 'select'),
                'order_by': payload.get('order_by', 'menu_order'),
                'has_archives': bool(payload.get('has_archives', False)),
                'terms': {},
            }
            store['attributes'][str(attribute_id)] = attribute
            return 201, _attribute_payload(attribute)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_attribute_payload(attribute) for attribute in store['attributes'].values()]
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def product_attribute_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    attribute_id = _coerce_int(request.match_info['attribute_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        attribute = store['attributes'].get(str(attribute_id))
        if attribute is None:
            return 404, {'message': 'attribute not found'}
        if request.method == 'DELETE':
            store['attributes'].pop(str(attribute_id), None)
            return 200, {'deleted': True, 'previous': _attribute_payload(attribute)}
        if request.method == 'PUT':
            for key in ('name', 'slug', 'type', 'order_by'):
                if key in payload:
                    attribute[key] = payload[key]
            if 'has_archives' in payload:
                attribute['has_archives'] = bool(payload['has_archives'])
        return 200, _attribute_payload(attribute)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def attribute_terms_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    attribute_id = _coerce_int(request.match_info['attribute_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            attribute = store['attributes'].get(str(attribute_id))
            if attribute is None:
                return 404, {'message': 'attribute not found'}
            term_id = _next_id(state, 'term')
            name = str(payload.get('name') or f'Term {term_id}')
            term = {
                'id': term_id,
                'name': name,
                'slug': slugify(str(payload.get('slug') or name)),
                'description': str(payload.get('description') or ''),
                'menu_order': _coerce_int(payload.get('menu_order'), 0),
                'count': 0,
            }
            attribute['terms'][str(term_id)] = term
            return 201, _term_payload(term)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        store = _ensure_store(state, store_key)
        attribute = store['attributes'].get(str(attribute_id))
        if attribute is None:
            return 404, []
        rows = [_term_payload(term) for term in attribute.get('terms', {}).values()]
        rows.sort(key=lambda row: int(row['id']))
        return 200, _paginate(rows, request, default_per_page=100)

    status, response = mutate_store(_store_path(), _default_state, list_items)
    return _json(response, status=status)


async def product_variations_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    product_id = _coerce_int(request.match_info['product_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            product = store['products'].get(str(product_id))
            if product is None:
                return 404, {'message': 'product not found'}
            variation_id = _next_id(state, 'variation')
            image = payload.get('image') or {}
            media = store['media'].get(str(_coerce_int(image.get('id'))))
            variation = {
                'id': variation_id,
                'product_id': product_id,
                'regular_price': str(payload.get('regular_price') or payload.get('price') or ''),
                'price': str(payload.get('price') or payload.get('regular_price') or ''),
                'stock_quantity': payload.get('stock_quantity'),
                'manage_stock': bool(payload.get('manage_stock', False)),
                'stock_status': payload.get('stock_status', 'instock'),
                'attributes': list(payload.get('attributes') or []),
                'image': {
                    'id': media['id'],
                    'src': media.get('source_url', ''),
                    'name': media.get('filename', ''),
                    'alt': media.get('alt_text', ''),
                }
                if media
                else {},
                'meta_data': list(payload.get('meta_data') or []),
                'date_created': payload.get('date_created', utc_now_iso()),
                'sku': payload.get('sku', ''),
            }
            product['variations'][str(variation_id)] = variation
            return 201, _variation_payload(store, variation)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        store = _ensure_store(state, store_key)
        product = store['products'].get(str(product_id))
        if product is None:
            return 404, []
        rows = [_variation_payload(store, variation) for variation in product.get('variations', {}).values()]
        rows.sort(key=lambda row: int(row['id']))
        return 200, _paginate(rows, request, default_per_page=100)

    status, response = mutate_store(_store_path(), _default_state, list_items)
    return _json(response, status=status)


async def product_variation_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    product_id = _coerce_int(request.match_info['product_id'])
    variation_id = _coerce_int(request.match_info['variation_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        product = store['products'].get(str(product_id))
        if product is None:
            return 404, {'message': 'product not found'}
        variation = product.get('variations', {}).get(str(variation_id))
        if variation is None:
            return 404, {'message': 'variation not found'}
        if request.method == 'DELETE':
            product['variations'].pop(str(variation_id), None)
            return 200, {'deleted': True, 'previous': _variation_payload(store, variation)}
        if request.method == 'PUT':
            for key in ('regular_price', 'price', 'stock_quantity', 'stock_status', 'sku'):
                if key in payload:
                    variation[key] = payload[key]
            if 'manage_stock' in payload:
                variation['manage_stock'] = bool(payload['manage_stock'])
            if 'attributes' in payload:
                variation['attributes'] = list(payload.get('attributes') or [])
            if 'meta_data' in payload:
                variation['meta_data'] = _merge_meta_data(
                    list(variation.get('meta_data', [])), list(payload.get('meta_data') or [])
                )
            if 'image' in payload:
                image = payload.get('image') or {}
                media = store['media'].get(str(_coerce_int(image.get('id'))))
                variation['image'] = (
                    {
                        'id': media['id'],
                        'src': media.get('source_url', ''),
                        'name': media.get('filename', ''),
                        'alt': media.get('alt_text', ''),
                    }
                    if media
                    else {}
                )
        return 200, _variation_payload(store, variation)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def shipping_zones_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            zone_id = _next_id(state, 'shipping_zone')
            zone = {
                'id': zone_id,
                'name': str(payload.get('name') or f'Shipping Zone {zone_id}'),
                'order': _coerce_int(payload.get('order'), zone_id),
                'locations': list(payload.get('locations') or []),
                'methods': {},
            }
            store['shipping_zones'][str(zone_id)] = zone
            return 201, _shipping_zone_payload(zone)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_shipping_zone_payload(zone) for zone in store['shipping_zones'].values()]
        search = str(request.query.get('search') or '').strip()
        if search:
            rows = [row for row in rows if _match_search([row.get('name', '')], search)]
        rows.sort(key=lambda row: (_coerce_int(row.get('order')), _coerce_int(row.get('id'))))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def shipping_zone_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    zone_id = _coerce_int(request.match_info['zone_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        zone = store['shipping_zones'].get(str(zone_id))
        if zone is None:
            return 404, {'message': 'shipping zone not found'}
        if request.method == 'DELETE':
            if zone_id == 0:
                return 400, {'message': 'default shipping zone cannot be deleted'}
            store['shipping_zones'].pop(str(zone_id), None)
            return 200, {'deleted': True, 'previous': _shipping_zone_payload(zone)}
        if request.method == 'PUT':
            for key in ('name', 'locations'):
                if key in payload:
                    zone[key] = payload[key]
            if 'order' in payload:
                zone['order'] = _coerce_int(payload.get('order'), zone.get('order', 0))
        return 200, _shipping_zone_payload(zone)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def shipping_zone_methods_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    zone_id = _coerce_int(request.match_info['zone_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            zone = store['shipping_zones'].get(str(zone_id))
            if zone is None:
                return 404, {'message': 'shipping zone not found'}
            method_instance_id = _next_id(state, 'shipping_method')
            raw_method_id = str(payload.get('method_id') or payload.get('id') or 'flat_rate')
            method = {
                'id': method_instance_id,
                'instance_id': method_instance_id,
                'method_id': raw_method_id,
                'method_title': str(payload.get('method_title') or raw_method_id.replace('_', ' ').title()),
                'title': str(
                    payload.get('title') or payload.get('method_title') or raw_method_id.replace('_', ' ').title()
                ),
                'order': _coerce_int(payload.get('order'), method_instance_id),
                'enabled': _boolish(payload.get('enabled'), True),
                'settings': dict(payload.get('settings') or {}),
            }
            zone.setdefault('methods', {})[str(method_instance_id)] = method
            return 201, _shipping_zone_method_payload(method)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        store = _ensure_store(state, store_key)
        zone = store['shipping_zones'].get(str(zone_id))
        if zone is None:
            return 404, []
        rows = [_shipping_zone_method_payload(method) for method in zone.get('methods', {}).values()]
        rows.sort(key=lambda row: (_coerce_int(row.get('order')), _coerce_int(row.get('id'))))
        return 200, _paginate(rows, request, default_per_page=100)

    status, response = mutate_store(_store_path(), _default_state, list_items)
    return _json(response, status=status)


async def shipping_zone_method_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    zone_id = _coerce_int(request.match_info['zone_id'])
    instance_id = _coerce_int(request.match_info['instance_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        zone = store['shipping_zones'].get(str(zone_id))
        if zone is None:
            return 404, {'message': 'shipping zone not found'}
        methods = zone.setdefault('methods', {})
        method = methods.get(str(instance_id))
        if method is None:
            return 404, {'message': 'shipping zone method not found'}
        if request.method == 'DELETE':
            methods.pop(str(instance_id), None)
            return 200, {'deleted': True, 'previous': _shipping_zone_method_payload(method)}
        if request.method == 'PUT':
            for key in ('title', 'method_title', 'method_id'):
                if key in payload:
                    method[key] = payload[key]
            if 'order' in payload:
                method['order'] = _coerce_int(payload.get('order'), method.get('order', 0))
            if 'enabled' in payload:
                method['enabled'] = _boolish(payload.get('enabled'), True)
            if 'settings' in payload:
                method['settings'] = dict(payload.get('settings') or {})
        return 200, _shipping_zone_method_payload(method)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def taxes_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            tax_id = _next_id(state, 'tax')
            tax = {
                'id': tax_id,
                'country': str(payload.get('country') or ''),
                'state': str(payload.get('state') or ''),
                'postcode': str(payload.get('postcode') or ''),
                'city': str(payload.get('city') or ''),
                'rate': str(payload.get('rate') or '0'),
                'name': str(payload.get('name') or f'Tax {tax_id}'),
                'priority': _coerce_int(payload.get('priority'), 1),
                'compound': _boolish(payload.get('compound'), False),
                'shipping': _boolish(payload.get('shipping'), True),
                'order': _coerce_int(payload.get('order'), tax_id),
                'class': str(payload.get('class') or ''),
            }
            store['taxes'][str(tax_id)] = tax
            return 201, _tax_payload(tax)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_tax_payload(tax) for tax in store['taxes'].values()]
        search = str(request.query.get('search') or '').strip()
        if search:
            rows = [
                row
                for row in rows
                if _match_search(
                    [
                        row.get('name', ''),
                        row.get('country', ''),
                        row.get('state', ''),
                        row.get('postcode', ''),
                        row.get('city', ''),
                    ],
                    search,
                )
            ]
        rows.sort(key=lambda row: (_coerce_int(row.get('order')), _coerce_int(row.get('id'))))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def tax_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    tax_id = _coerce_int(request.match_info['tax_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        tax = store['taxes'].get(str(tax_id))
        if tax is None:
            return 404, {'message': 'tax not found'}
        if request.method == 'DELETE':
            store['taxes'].pop(str(tax_id), None)
            return 200, {'deleted': True, 'previous': _tax_payload(tax)}
        if request.method == 'PUT':
            for key in ('country', 'state', 'postcode', 'city', 'rate', 'name', 'class'):
                if key in payload:
                    tax[key] = payload[key]
            if 'priority' in payload:
                tax['priority'] = _coerce_int(payload.get('priority'), tax.get('priority', 1))
            if 'compound' in payload:
                tax['compound'] = _boolish(payload.get('compound'), False)
            if 'shipping' in payload:
                tax['shipping'] = _boolish(payload.get('shipping'), True)
            if 'order' in payload:
                tax['order'] = _coerce_int(payload.get('order'), tax.get('order', 0))
        return 200, _tax_payload(tax)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def posts_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            post_id = _next_id(state, 'post')
            title = _rendered_text(payload.get('title')) or f'Post {post_id}'
            now = utc_now_iso()
            post = {
                'id': post_id,
                'date': str(payload.get('date') or now),
                'date_gmt': str(payload.get('date_gmt') or payload.get('date') or now),
                'modified': str(payload.get('modified') or payload.get('date') or now),
                'modified_gmt': str(
                    payload.get('modified_gmt') or payload.get('modified') or payload.get('date') or now
                ),
                'slug': str(payload.get('slug') or slugify(title)),
                'status': str(payload.get('status') or 'publish'),
                'title': title,
                'content': _rendered_text(payload.get('content')),
                'excerpt': _rendered_text(payload.get('excerpt')),
                'author': _coerce_int(payload.get('author'), 1),
                'featured_media': _coerce_int(payload.get('featured_media'), 0),
                'comment_status': str(payload.get('comment_status') or 'open'),
                'ping_status': str(payload.get('ping_status') or 'open'),
                'sticky': _boolish(payload.get('sticky'), False),
                'template': str(payload.get('template') or ''),
                'format': str(payload.get('format') or 'standard'),
                'meta': list(payload.get('meta') or []),
                'categories': list(payload.get('categories') or []),
                'tags': list(payload.get('tags') or []),
            }
            store['posts'][str(post_id)] = post
            return 201, _post_payload(request, store_key, post)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_post_payload(request, store_key, post) for post in store['posts'].values()]
        status_filter = str(request.query.get('status') or 'publish').strip().lower()
        search = str(request.query.get('search') or '').strip()
        if status_filter and status_filter != 'any':
            rows = [row for row in rows if str(row.get('status') or '').lower() == status_filter]
        if search:
            rows = [
                row
                for row in rows
                if _match_search(
                    [
                        row.get('slug', ''),
                        row.get('title', {}).get('rendered', ''),
                        row.get('content', {}).get('rendered', ''),
                    ],
                    search,
                )
            ]
        orderby = str(request.query.get('orderby') or 'date').strip().lower()
        reverse = str(request.query.get('order') or 'desc').strip().lower() != 'asc'
        if orderby == 'date':
            rows.sort(key=lambda row: (str(row.get('date') or ''), _coerce_int(row.get('id'))), reverse=reverse)
        else:
            rows.sort(key=lambda row: _coerce_int(row.get('id')), reverse=reverse)
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def post_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    post_id = _coerce_int(request.match_info['post_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        post = store['posts'].get(str(post_id))
        if post is None:
            return 404, {'message': 'post not found'}
        if request.method == 'DELETE':
            store['posts'].pop(str(post_id), None)
            return 200, {'deleted': True, 'previous': _post_payload(request, store_key, post)}
        if request.method == 'PUT':
            if 'title' in payload:
                post['title'] = _rendered_text(payload.get('title'))
            if 'content' in payload:
                post['content'] = _rendered_text(payload.get('content'))
            if 'excerpt' in payload:
                post['excerpt'] = _rendered_text(payload.get('excerpt'))
            for key in ('date', 'date_gmt', 'slug', 'status', 'comment_status', 'ping_status', 'template', 'format'):
                if key in payload:
                    post[key] = str(payload.get(key) or '')
            for key in ('meta', 'categories', 'tags'):
                if key in payload:
                    post[key] = list(payload.get(key) or [])
            if 'author' in payload:
                post['author'] = _coerce_int(payload.get('author'), 1)
            if 'featured_media' in payload:
                post['featured_media'] = _coerce_int(payload.get('featured_media'), 0)
            if 'sticky' in payload:
                post['sticky'] = _boolish(payload.get('sticky'), False)
            post['modified'] = utc_now_iso()
            post['modified_gmt'] = post['modified']
        return 200, _post_payload(request, store_key, post)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def customers_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            customer_id = _next_id(state, 'customer')
            email = str(payload.get('email') or f'customer{customer_id}@woocommerce.mock')
            customer = {
                'id': customer_id,
                'email': email,
                'first_name': payload.get('first_name', ''),
                'last_name': payload.get('last_name', ''),
                'role': payload.get('role', 'customer'),
                'username': payload.get('username') or email.split('@', 1)[0],
                'date_created': utc_now_iso(),
                'billing': dict(payload.get('billing') or {'email': email}),
                'shipping': dict(payload.get('shipping') or {}),
                'meta_data': list(payload.get('meta_data') or []),
            }
            store['customers'][str(customer_id)] = customer
            return 201, _customer_payload(customer)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_customer_payload(customer) for customer in store['customers'].values()]
        search = str(request.query.get('search') or '').strip()
        email = str(request.query.get('email') or '').strip()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if email and str(row.get('email') or '').lower() != email.lower():
                continue
            if search and not _match_search(
                [row.get('email', ''), row.get('first_name', ''), row.get('last_name', ''), row.get('username', '')],
                search,
            ):
                continue
            filtered.append(row)
        filtered.sort(key=lambda row: int(row['id']))
        return _paginate(filtered, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def customer_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    customer_id = _coerce_int(request.match_info['customer_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        customer = store['customers'].get(str(customer_id))
        if customer is None:
            return 404, {'message': 'customer not found'}
        if request.method == 'DELETE':
            store['customers'].pop(str(customer_id), None)
            return 200, {'deleted': True, 'previous': _customer_payload(customer)}
        if request.method == 'PUT':
            for key in ('email', 'first_name', 'last_name', 'role', 'username'):
                if key in payload:
                    customer[key] = payload[key]
            if 'billing' in payload:
                customer['billing'] = dict(payload.get('billing') or {})
            if 'shipping' in payload:
                customer['shipping'] = dict(payload.get('shipping') or {})
            if 'meta_data' in payload:
                customer['meta_data'] = _merge_meta_data(
                    list(customer.get('meta_data', [])), list(payload.get('meta_data') or [])
                )
        return 200, _customer_payload(customer)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def customers_batch(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    payload = await _request_json(request)

    def handle(state: dict[str, Any]) -> dict[str, Any]:
        store = _ensure_store(state, store_key)
        created: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []
        for item in payload.get('create') or []:
            customer_id = _next_id(state, 'customer')
            email = str(item.get('email') or f'customer{customer_id}@woocommerce.mock')
            customer = {
                'id': customer_id,
                'email': email,
                'first_name': item.get('first_name', ''),
                'last_name': item.get('last_name', ''),
                'role': item.get('role', 'customer'),
                'username': item.get('username') or email.split('@', 1)[0],
                'date_created': utc_now_iso(),
                'billing': dict(item.get('billing') or {'email': email}),
                'shipping': dict(item.get('shipping') or {}),
                'meta_data': list(item.get('meta_data') or []),
            }
            store['customers'][str(customer_id)] = customer
            created.append(_customer_payload(customer))
        for item in payload.get('update') or []:
            customer_id = _coerce_int(item.get('id'))
            customer = store['customers'].get(str(customer_id))
            if customer is None:
                continue
            for key in ('email', 'first_name', 'last_name', 'role', 'username'):
                if key in item:
                    customer[key] = item[key]
            if 'billing' in item:
                customer['billing'] = dict(item.get('billing') or {})
            if 'shipping' in item:
                customer['shipping'] = dict(item.get('shipping') or {})
            if 'meta_data' in item:
                customer['meta_data'] = _merge_meta_data(
                    list(customer.get('meta_data', [])), list(item.get('meta_data') or [])
                )
            updated.append(_customer_payload(customer))
        for item in payload.get('delete') or []:
            customer_id = _coerce_int(item)
            if not customer_id or customer_id == DEFAULT_CUSTOMER_ID:
                continue
            if store['customers'].pop(str(customer_id), None) is not None:
                deleted.append({'id': customer_id})
        return {'create': created, 'update': updated, 'delete': deleted}

    return _json(mutate_store(_store_path(), _default_state, handle))


async def customer_downloads(request: web.Request) -> web.Response:
    return _json([])


async def orders_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            order_id = _next_id(state, 'order')
            customer = _resolve_customer(store, state, payload)
            line_items = list(payload.get('line_items') or [])
            order = {
                'id': order_id,
                'status': payload.get('status', 'pending'),
                'customer_id': customer['id'],
                'payment_method': payload.get('payment_method', ''),
                'payment_method_title': payload.get('payment_method_title', ''),
                'billing': dict(payload.get('billing') or customer.get('billing', {})),
                'shipping': dict(payload.get('shipping') or customer.get('shipping', {})),
                'line_items': line_items,
                'meta_data': list(payload.get('meta_data') or []),
                'date_created': payload.get('date_created', utc_now_iso()),
                'date_modified': utc_now_iso(),
                'notes': [],
                'refunds': [],
            }
            store['orders'][str(order_id)] = order
            return 201, _order_payload(store, order)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_order_payload(store, order) for order in store['orders'].values()]
        customer_id = _coerce_int(request.query.get('customer'))
        status_filter = str(request.query.get('status') or '').strip()
        search = str(request.query.get('search') or '').strip()
        after = str(request.query.get('after') or '').strip()
        before = str(request.query.get('before') or '').strip()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if customer_id and _coerce_int(row.get('customer_id')) != customer_id:
                continue
            if status_filter and str(row.get('status') or '') != status_filter:
                continue
            created = str(row.get('date_created') or '')
            if after and created and created < after:
                continue
            if before and created and created > before:
                continue
            if search and not _match_search(
                [
                    str(row.get('id')),
                    str(row.get('billing', {}).get('email', '')),
                    str(row.get('billing', {}).get('first_name', '')),
                    str(row.get('billing', {}).get('last_name', '')),
                ],
                search,
            ):
                continue
            filtered.append(row)
        filtered.sort(key=lambda row: int(row['id']))
        return _paginate(filtered, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def order_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    order_id = _coerce_int(request.match_info['order_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        order = store['orders'].get(str(order_id))
        if order is None:
            return 404, {'message': 'order not found'}
        if request.method == 'DELETE':
            store['orders'].pop(str(order_id), None)
            return 200, {'deleted': True, 'previous': _order_payload(store, order)}
        if request.method == 'PUT':
            for key in ('status', 'payment_method', 'payment_method_title'):
                if key in payload:
                    order[key] = payload[key]
            if 'billing' in payload:
                order['billing'] = dict(payload.get('billing') or {})
            if 'shipping' in payload:
                order['shipping'] = dict(payload.get('shipping') or {})
            if 'line_items' in payload:
                order['line_items'] = list(payload.get('line_items') or [])
            if 'meta_data' in payload:
                order['meta_data'] = _merge_meta_data(
                    list(order.get('meta_data', [])), list(payload.get('meta_data') or [])
                )
            order['date_modified'] = utc_now_iso()
        return 200, _order_payload(store, order)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def orders_batch(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    payload = await _request_json(request)

    def handle(state: dict[str, Any]) -> dict[str, Any]:
        store = _ensure_store(state, store_key)
        created: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []
        for item in payload.get('create') or []:
            order_id = _next_id(state, 'order')
            customer = _resolve_customer(store, state, item)
            order = {
                'id': order_id,
                'status': item.get('status', 'pending'),
                'customer_id': customer['id'],
                'payment_method': item.get('payment_method', ''),
                'payment_method_title': item.get('payment_method_title', ''),
                'billing': dict(item.get('billing') or customer.get('billing', {})),
                'shipping': dict(item.get('shipping') or customer.get('shipping', {})),
                'line_items': list(item.get('line_items') or []),
                'meta_data': list(item.get('meta_data') or []),
                'date_created': item.get('date_created', utc_now_iso()),
                'date_modified': utc_now_iso(),
                'notes': [],
                'refunds': [],
            }
            store['orders'][str(order_id)] = order
            created.append(_order_payload(store, order))
        for item in payload.get('update') or []:
            order_id = _coerce_int(item.get('id'))
            order = store['orders'].get(str(order_id))
            if order is None:
                continue
            for key in ('status', 'payment_method', 'payment_method_title'):
                if key in item:
                    order[key] = item[key]
            if 'billing' in item:
                order['billing'] = dict(item.get('billing') or {})
            if 'shipping' in item:
                order['shipping'] = dict(item.get('shipping') or {})
            if 'line_items' in item:
                order['line_items'] = list(item.get('line_items') or [])
            if 'meta_data' in item:
                order['meta_data'] = _merge_meta_data(
                    list(order.get('meta_data', [])), list(item.get('meta_data') or [])
                )
            order['date_modified'] = utc_now_iso()
            updated.append(_order_payload(store, order))
        for item in payload.get('delete') or []:
            order_id = _coerce_int(item)
            if not order_id:
                continue
            if store['orders'].pop(str(order_id), None) is not None:
                deleted.append({'id': order_id})
        return {'create': created, 'update': updated, 'delete': deleted}

    return _json(mutate_store(_store_path(), _default_state, handle))


async def order_notes_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    order_id = _coerce_int(request.match_info['order_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            order = store['orders'].get(str(order_id))
            if order is None:
                return 404, {'message': 'order not found'}
            note = {
                'id': _next_id(state, 'note'),
                'note': payload.get('note', ''),
                'customer_note': bool(payload.get('customer_note', False)),
                'date_created': utc_now_iso(),
            }
            order.setdefault('notes', []).append(note)
            return 201, dict(note)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        store = _ensure_store(state, store_key)
        order = store['orders'].get(str(order_id))
        if order is None:
            return 404, []
        return 200, list(order.get('notes', []))

    status, response = mutate_store(_store_path(), _default_state, list_items)
    return _json(response, status=status)


async def order_note_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    order_id = _coerce_int(request.match_info['order_id'])
    note_id = _coerce_int(request.match_info['note_id'])

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        order = store['orders'].get(str(order_id))
        if order is None:
            return 404, {'message': 'order not found'}
        notes = order.get('notes', [])
        for index, note in enumerate(notes):
            if _coerce_int(note.get('id')) == note_id:
                if request.method == 'DELETE':
                    removed = notes.pop(index)
                    return 200, {'deleted': True, 'previous': removed}
                return 200, dict(note)
        return 404, {'message': 'note not found'}

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def order_refunds_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    order_id = _coerce_int(request.match_info['order_id'])
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            order = store['orders'].get(str(order_id))
            if order is None:
                return 404, {'message': 'order not found'}
            refund = {
                'id': _next_id(state, 'refund'),
                'amount': str(payload.get('amount', '0')),
                'reason': payload.get('reason', ''),
                'date_created': utc_now_iso(),
            }
            order.setdefault('refunds', []).append(refund)
            return 201, dict(refund)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        store = _ensure_store(state, store_key)
        order = store['orders'].get(str(order_id))
        if order is None:
            return 404, []
        return 200, list(order.get('refunds', []))

    status, response = mutate_store(_store_path(), _default_state, list_items)
    return _json(response, status=status)


async def order_refund_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    order_id = _coerce_int(request.match_info['order_id'])
    refund_id = _coerce_int(request.match_info['refund_id'])

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        order = store['orders'].get(str(order_id))
        if order is None:
            return 404, {'message': 'order not found'}
        refunds = order.get('refunds', [])
        for index, refund in enumerate(refunds):
            if _coerce_int(refund.get('id')) == refund_id:
                if request.method == 'DELETE':
                    removed = refunds.pop(index)
                    return 200, {'deleted': True, 'previous': removed}
                return 200, dict(refund)
        return 404, {'message': 'refund not found'}

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def coupons_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            coupon_id = _next_id(state, 'coupon')
            coupon = {
                'id': coupon_id,
                'code': payload.get('code', f'coupon-{coupon_id}'),
                'description': payload.get('description', ''),
                'amount': payload.get('amount', '0'),
                'discount_type': payload.get('discount_type', 'fixed_cart'),
                'meta_data': list(payload.get('meta_data') or []),
                'date_created': utc_now_iso(),
            }
            store['coupons'][str(coupon_id)] = coupon
            return 201, _coupon_payload(coupon)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [_coupon_payload(coupon) for coupon in store['coupons'].values()]
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def coupon_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    coupon_id = _coerce_int(request.match_info['coupon_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        coupon = store['coupons'].get(str(coupon_id))
        if coupon is None:
            return 404, {'message': 'coupon not found'}
        if request.method == 'DELETE':
            store['coupons'].pop(str(coupon_id), None)
            return 200, {'deleted': True, 'previous': _coupon_payload(coupon)}
        if request.method == 'PUT':
            for key in ('code', 'description', 'amount', 'discount_type'):
                if key in payload:
                    coupon[key] = payload[key]
            if 'meta_data' in payload:
                coupon['meta_data'] = _merge_meta_data(
                    list(coupon.get('meta_data', [])), list(payload.get('meta_data') or [])
                )
        return 200, _coupon_payload(coupon)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def coupons_batch(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    payload = await _request_json(request)

    def handle(state: dict[str, Any]) -> dict[str, Any]:
        store = _ensure_store(state, store_key)
        created: list[dict[str, Any]] = []
        updated: list[dict[str, Any]] = []
        deleted: list[dict[str, Any]] = []
        for item in payload.get('create') or []:
            coupon_id = _next_id(state, 'coupon')
            coupon = {
                'id': coupon_id,
                'code': item.get('code', f'coupon-{coupon_id}'),
                'description': item.get('description', ''),
                'amount': item.get('amount', '0'),
                'discount_type': item.get('discount_type', 'fixed_cart'),
                'meta_data': list(item.get('meta_data') or []),
                'date_created': utc_now_iso(),
            }
            store['coupons'][str(coupon_id)] = coupon
            created.append(_coupon_payload(coupon))
        for item in payload.get('update') or []:
            coupon_id = _coerce_int(item.get('id'))
            coupon = store['coupons'].get(str(coupon_id))
            if coupon is None:
                continue
            for key in ('code', 'description', 'amount', 'discount_type'):
                if key in item:
                    coupon[key] = item[key]
            if 'meta_data' in item:
                coupon['meta_data'] = _merge_meta_data(
                    list(coupon.get('meta_data', [])), list(item.get('meta_data') or [])
                )
            updated.append(_coupon_payload(coupon))
        for item in payload.get('delete') or []:
            coupon_id = _coerce_int(item)
            if not coupon_id:
                continue
            if store['coupons'].pop(str(coupon_id), None) is not None:
                deleted.append({'id': coupon_id})
        return {'create': created, 'update': updated, 'delete': deleted}

    return _json(mutate_store(_store_path(), _default_state, handle))


async def product_reviews_collection(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    if request.method == 'POST':
        payload = await _request_json(request)

        def create(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            store = _ensure_store(state, store_key)
            review_id = _next_id(state, 'review')
            review = {
                'id': review_id,
                'product_id': _coerce_int(payload.get('product_id')),
                'review': payload.get('review', ''),
                'reviewer': payload.get('reviewer', ''),
                'reviewer_email': payload.get('reviewer_email', ''),
                'rating': _coerce_int(payload.get('rating'), 0),
                'date_created': utc_now_iso(),
            }
            store['reviews'][str(review_id)] = review
            return 201, dict(review)

        status, response = mutate_store(_store_path(), _default_state, create)
        return _json(response, status=status)

    def list_items(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = [dict(review) for review in store['reviews'].values()]
        rows.sort(key=lambda row: int(row['id']))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, list_items))


async def product_review_item(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    review_id = _coerce_int(request.match_info['review_id'])
    payload = await _request_json(request) if request.method == 'PUT' else {}

    def handle(state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        store = _ensure_store(state, store_key)
        review = store['reviews'].get(str(review_id))
        if review is None:
            return 404, {'message': 'review not found'}
        if request.method == 'DELETE':
            store['reviews'].pop(str(review_id), None)
            return 200, {'deleted': True, 'previous': dict(review)}
        if request.method == 'PUT':
            for key in ('review', 'reviewer', 'reviewer_email', 'rating'):
                if key in payload:
                    review[key] = payload[key]
        return 200, dict(review)

    status, response = mutate_store(_store_path(), _default_state, handle)
    return _json(response, status=status)


async def reports_sales(request: web.Request) -> web.Response:
    store_key = _store_key(request)

    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        total_orders = 0
        total_sales = 0.0
        for order in store['orders'].values():
            if str(order.get('status') or '').lower() in {'cancelled', 'failed', 'trash'}:
                continue
            total_orders += 1
            for item in _order_line_items(store, list(order.get('line_items', []))):
                price = float(item.get('price') or 0)
                quantity = _coerce_int(item.get('quantity'), 0)
                total_sales += price * quantity
        return [
            {'total_sales': round(total_sales, 2), 'net_sales': round(total_sales, 2), 'total_orders': total_orders}
        ]

    return _json(mutate_store(_store_path(), _default_state, build))


async def reports_top_sellers(request: web.Request) -> web.Response:
    store_key = _store_key(request)

    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        product_totals, _variation_totals = _product_sales(store)
        rows: list[dict[str, Any]] = []
        for product_id, quantity in product_totals.items():
            product = store['products'].get(str(product_id))
            if product is None:
                continue
            rows.append({'product_id': product_id, 'title': product.get('name', ''), 'quantity': quantity})
        rows.sort(key=lambda row: (-_coerce_int(row.get('quantity')), _coerce_int(row.get('product_id'))))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, build))


async def reports_counts(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    path = request.path.rstrip('/')
    parts = [part for part in path.split('/') if part]
    target = parts[-2] if len(parts) >= 2 else ''

    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        if target == 'customers':
            return [{'slug': 'all', 'name': 'All', 'total': len(store['customers'])}]
        if target == 'products':
            return [{'slug': 'all', 'name': 'All', 'total': len(store['products'])}]
        if target == 'coupons':
            return [{'slug': 'all', 'name': 'All', 'total': len(store['coupons'])}]
        if target == 'reviews':
            return [{'slug': 'all', 'name': 'All', 'total': len(store['reviews'])}]
        totals: dict[str, int] = {}
        for order in store['orders'].values():
            status = str(order.get('status') or 'unknown')
            totals[status] = totals.get(status, 0) + 1
        rows = [{'slug': status, 'name': status, 'total': total} for status, total in sorted(totals.items())]
        if not rows:
            rows.append({'slug': 'all', 'name': 'All', 'total': 0})
        return rows

    return _json(mutate_store(_store_path(), _default_state, build))


async def reports_stock(request: web.Request) -> web.Response:
    store_key = _store_key(request)
    low_only = request.path.rstrip('/').endswith('/low')

    def build(state: dict[str, Any]) -> list[dict[str, Any]]:
        store = _ensure_store(state, store_key)
        rows = []
        for product in store['products'].values():
            payload = _product_payload(store, product)
            stock_quantity = _coerce_int(payload.get('stock_quantity'))
            if low_only and stock_quantity > 5:
                continue
            rows.append({
                'product_id': payload['id'],
                'name': payload.get('name', ''),
                'stock_quantity': payload.get('stock_quantity'),
                'stock_status': payload.get('stock_status', 'instock'),
            })
        rows.sort(key=lambda row: _coerce_int(row.get('product_id')))
        return _paginate(rows, request, default_per_page=100)

    return _json(mutate_store(_store_path(), _default_state, build))


def _register_variant_routes(app: web.Application, method: str, path: str, handler: Any) -> None:
    add = getattr(app.router, f'add_{method.lower()}')
    add(path, handler)
    add(f'/{{store:[^/]+}}{path}', handler)


def _build_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_get('/healthz', health)
    app.router.add_get('/', site_root)
    app.router.add_get('/{store:[^/]+}', site_root)

    for method, path, handler in [
        ('GET', '/wp-login.php', wp_login_get),
        ('POST', '/wp-login.php', wp_login_post),
        ('GET', '/wp-admin/admin-ajax.php', wp_rest_nonce),
        ('GET', '/wp-json/wp/v2/media', media_collection),
        ('POST', '/wp-json/wp/v2/media', media_collection),
        ('GET', '/wp-json/wp/v2/media/{media_id}', media_item),
        ('DELETE', '/wp-json/wp/v2/media/{media_id}', media_item),
        ('GET', '/mock-media/{media_id}/{filename}', media_file),
        ('GET', '/wp-json/wc/v3/system_status', system_status),
        ('GET', '/wp-json/wc/v3/products', products_collection),
        ('POST', '/wp-json/wc/v3/products', products_collection),
        ('POST', '/wp-json/wc/v3/products/batch', products_batch),
        ('GET', '/wp-json/wc/v3/products/categories', product_categories_collection),
        ('POST', '/wp-json/wc/v3/products/categories', product_categories_collection),
        ('POST', '/wp-json/wc/v3/products/categories/batch', product_categories_batch),
        ('GET', '/wp-json/wc/v3/products/categories/{category_id}', product_category_item),
        ('PUT', '/wp-json/wc/v3/products/categories/{category_id}', product_category_item),
        ('DELETE', '/wp-json/wc/v3/products/categories/{category_id}', product_category_item),
        ('GET', '/wp-json/wc/v3/products/tags', product_tags_collection),
        ('POST', '/wp-json/wc/v3/products/tags', product_tags_collection),
        ('GET', '/wp-json/wc/v3/products/attributes', product_attributes_collection),
        ('POST', '/wp-json/wc/v3/products/attributes', product_attributes_collection),
        ('GET', '/wp-json/wc/v3/products/attributes/{attribute_id}', product_attribute_item),
        ('PUT', '/wp-json/wc/v3/products/attributes/{attribute_id}', product_attribute_item),
        ('DELETE', '/wp-json/wc/v3/products/attributes/{attribute_id}', product_attribute_item),
        ('GET', '/wp-json/wc/v3/products/attributes/{attribute_id}/terms', attribute_terms_collection),
        ('POST', '/wp-json/wc/v3/products/attributes/{attribute_id}/terms', attribute_terms_collection),
        ('GET', '/wp-json/wc/v3/products/{product_id}', product_item),
        ('PUT', '/wp-json/wc/v3/products/{product_id}', product_item),
        ('DELETE', '/wp-json/wc/v3/products/{product_id}', product_item),
        ('GET', '/wp-json/wc/v3/products/{product_id}/variations', product_variations_collection),
        ('POST', '/wp-json/wc/v3/products/{product_id}/variations', product_variations_collection),
        ('GET', '/wp-json/wc/v3/products/{product_id}/variations/{variation_id}', product_variation_item),
        ('PUT', '/wp-json/wc/v3/products/{product_id}/variations/{variation_id}', product_variation_item),
        ('DELETE', '/wp-json/wc/v3/products/{product_id}/variations/{variation_id}', product_variation_item),
        ('GET', '/wp-json/wc/v3/shipping/zones', shipping_zones_collection),
        ('POST', '/wp-json/wc/v3/shipping/zones', shipping_zones_collection),
        ('GET', '/wp-json/wc/v3/shipping/zones/{zone_id}', shipping_zone_item),
        ('PUT', '/wp-json/wc/v3/shipping/zones/{zone_id}', shipping_zone_item),
        ('DELETE', '/wp-json/wc/v3/shipping/zones/{zone_id}', shipping_zone_item),
        ('GET', '/wp-json/wc/v3/shipping/zones/{zone_id}/methods', shipping_zone_methods_collection),
        ('POST', '/wp-json/wc/v3/shipping/zones/{zone_id}/methods', shipping_zone_methods_collection),
        ('GET', '/wp-json/wc/v3/shipping/zones/{zone_id}/methods/{instance_id}', shipping_zone_method_item),
        ('PUT', '/wp-json/wc/v3/shipping/zones/{zone_id}/methods/{instance_id}', shipping_zone_method_item),
        ('DELETE', '/wp-json/wc/v3/shipping/zones/{zone_id}/methods/{instance_id}', shipping_zone_method_item),
        ('GET', '/wp-json/wc/v3/taxes', taxes_collection),
        ('POST', '/wp-json/wc/v3/taxes', taxes_collection),
        ('GET', '/wp-json/wc/v3/taxes/{tax_id}', tax_item),
        ('PUT', '/wp-json/wc/v3/taxes/{tax_id}', tax_item),
        ('DELETE', '/wp-json/wc/v3/taxes/{tax_id}', tax_item),
        ('GET', '/wp-json/wp/v2/posts', posts_collection),
        ('POST', '/wp-json/wp/v2/posts', posts_collection),
        ('GET', '/wp-json/wp/v2/posts/{post_id}', post_item),
        ('PUT', '/wp-json/wp/v2/posts/{post_id}', post_item),
        ('DELETE', '/wp-json/wp/v2/posts/{post_id}', post_item),
        ('GET', '/wp-json/wc/v3/customers', customers_collection),
        ('POST', '/wp-json/wc/v3/customers', customers_collection),
        ('POST', '/wp-json/wc/v3/customers/batch', customers_batch),
        ('GET', '/wp-json/wc/v3/customers/{customer_id}', customer_item),
        ('PUT', '/wp-json/wc/v3/customers/{customer_id}', customer_item),
        ('DELETE', '/wp-json/wc/v3/customers/{customer_id}', customer_item),
        ('GET', '/wp-json/wc/v3/customers/{customer_id}/downloads', customer_downloads),
        ('GET', '/wp-json/wc/v3/orders', orders_collection),
        ('POST', '/wp-json/wc/v3/orders', orders_collection),
        ('POST', '/wp-json/wc/v3/orders/batch', orders_batch),
        ('GET', '/wp-json/wc/v3/orders/{order_id}', order_item),
        ('PUT', '/wp-json/wc/v3/orders/{order_id}', order_item),
        ('DELETE', '/wp-json/wc/v3/orders/{order_id}', order_item),
        ('GET', '/wp-json/wc/v3/orders/{order_id}/notes', order_notes_collection),
        ('POST', '/wp-json/wc/v3/orders/{order_id}/notes', order_notes_collection),
        ('GET', '/wp-json/wc/v3/orders/{order_id}/notes/{note_id}', order_note_item),
        ('DELETE', '/wp-json/wc/v3/orders/{order_id}/notes/{note_id}', order_note_item),
        ('GET', '/wp-json/wc/v3/orders/{order_id}/refunds', order_refunds_collection),
        ('POST', '/wp-json/wc/v3/orders/{order_id}/refunds', order_refunds_collection),
        ('GET', '/wp-json/wc/v3/orders/{order_id}/refunds/{refund_id}', order_refund_item),
        ('DELETE', '/wp-json/wc/v3/orders/{order_id}/refunds/{refund_id}', order_refund_item),
        ('GET', '/wp-json/wc/v3/coupons', coupons_collection),
        ('POST', '/wp-json/wc/v3/coupons', coupons_collection),
        ('POST', '/wp-json/wc/v3/coupons/batch', coupons_batch),
        ('GET', '/wp-json/wc/v3/coupons/{coupon_id}', coupon_item),
        ('PUT', '/wp-json/wc/v3/coupons/{coupon_id}', coupon_item),
        ('DELETE', '/wp-json/wc/v3/coupons/{coupon_id}', coupon_item),
        ('GET', '/wp-json/wc/v3/products/reviews', product_reviews_collection),
        ('POST', '/wp-json/wc/v3/products/reviews', product_reviews_collection),
        ('GET', '/wp-json/wc/v3/products/reviews/{review_id}', product_review_item),
        ('PUT', '/wp-json/wc/v3/products/reviews/{review_id}', product_review_item),
        ('DELETE', '/wp-json/wc/v3/products/reviews/{review_id}', product_review_item),
        ('GET', '/wp-json/wc/v3/reports/sales', reports_sales),
        ('GET', '/wp-json/wc/v3/reports/top_sellers', reports_top_sellers),
        ('GET', '/wp-json/wc/v3/reports/coupons/totals', reports_counts),
        ('GET', '/wp-json/wc/v3/reports/customers/totals', reports_counts),
        ('GET', '/wp-json/wc/v3/reports/orders/totals', reports_counts),
        ('GET', '/wp-json/wc/v3/reports/products/totals', reports_counts),
        ('GET', '/wp-json/wc/v3/reports/reviews/totals', reports_counts),
        ('GET', '/wp-json/wc/v3/reports/stock', reports_stock),
        ('GET', '/wp-json/wc/v3/reports/stock/low', reports_stock),
    ]:
        _register_variant_routes(app, method, path, handler)
    return app


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    runner = web.AppRunner(_build_app())
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    while True:
        await __import__('asyncio').sleep(3600)


if __name__ == '__main__':
    __import__('asyncio').run(main())
