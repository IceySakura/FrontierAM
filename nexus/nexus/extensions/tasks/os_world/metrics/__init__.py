"""OSWorld evaluator metric registry."""

from __future__ import annotations

import importlib
import logging

from collections.abc import Callable
from typing import Any

import nexus.extensions.tasks.os_world.metrics.simple  # noqa: F401

from nexus.extensions.tasks.os_world.evaluation.registry import metric_registry


logger = logging.getLogger(__name__)
_SIMPLE_FALLBACKS = dict(metric_registry.items())


class OfficialMetricUnavailableError(NotImplementedError):
    """Raised when an official metric cannot load and no lightweight fallback exists."""


_OFFICIAL_METRICS: dict[str, tuple[str, str]] = {
    'check_gnome_favorite_apps': ('basic_os', 'check_gnome_favorite_apps'),
    'is_utc_0': ('basic_os', 'is_utc_0'),
    'check_text_enlarged': ('basic_os', 'check_text_enlarged'),
    'check_moved_jpgs': ('basic_os', 'check_moved_jpgs'),
    'is_in_vm_clickboard': ('basic_os', 'is_in_vm_clickboard'),
    'is_expected_tabs': ('chrome', 'is_expected_tabs'),
    'is_expected_bookmarks': ('chrome', 'is_expected_bookmarks'),
    'compare_pdfs': ('chrome', 'compare_pdfs'),
    'compare_htmls': ('chrome', 'compare_htmls'),
    'compare_archive': ('chrome', 'compare_archive'),
    'is_cookie_deleted': ('chrome', 'is_cookie_deleted'),
    'is_shortcut_on_desktop': ('chrome', 'is_shortcut_on_desktop'),
    'check_font_size': ('chrome', 'check_font_size'),
    'check_enabled_experiments': ('chrome', 'check_enabled_experiments'),
    'check_history_deleted': ('chrome', 'check_history_deleted'),
    'is_expected_search_query': ('chrome', 'is_expected_search_query'),
    'is_expected_active_tab': ('chrome', 'is_expected_active_tab'),
    'is_expected_url_pattern_match': ('chrome', 'is_expected_url_pattern_match'),
    'is_added_to_steam_cart': ('chrome', 'is_added_to_steam_cart'),
    'is_expected_installed_extensions': ('chrome', 'is_expected_installed_extensions'),
    'compare_pdf_images': ('chrome', 'compare_pdf_images'),
    'compare_pdf_form_fields': ('chrome', 'compare_pdf_form_fields'),
    'is_expected_active_tab_approximate': ('chrome', 'is_expected_active_tab_approximate'),
    'compare_font_names': ('docs', 'compare_font_names'),
    'compare_subscript_contains': ('docs', 'compare_subscript_contains'),
    'has_page_numbers_in_footers': ('docs', 'has_page_numbers_in_footers'),
    'compare_docx_lines': ('docs', 'compare_docx_lines'),
    'evaluate_colored_words_in_tables': ('docs', 'evaluate_colored_words_in_tables'),
    'check_highlighted_words': ('docs', 'check_highlighted_words'),
    'evaluate_strike_through_last_paragraph': ('docs', 'evaluate_strike_through_last_paragraph'),
    'evaluate_conversion': ('docs', 'evaluate_conversion'),
    'evaluate_spacing': ('docs', 'evaluate_spacing'),
    'check_italic_font_size_14': ('docs', 'check_italic_font_size_14'),
    'evaluate_alignment': ('docs', 'evaluate_alignment'),
    'get_unique_train_ids': ('docs', 'get_unique_train_ids'),
    'check_no_duplicates': ('docs', 'check_no_duplicates'),
    'compare_init_lines': ('docs', 'compare_init_lines'),
    'find_default_font': ('docs', 'find_default_font'),
    'contains_page_break': ('docs', 'contains_page_break'),
    'compare_docx_files': ('docs', 'compare_docx_files'),
    'compare_docx_tables': ('docs', 'compare_docx_tables'),
    'compare_line_spacing': ('docs', 'compare_line_spacing'),
    'compare_insert_equation': ('docs', 'compare_insert_equation'),
    'compare_highlighted_text': ('docs', 'compare_highlighted_text'),
    'is_first_line_centered': ('docs', 'is_first_line_centered'),
    'check_file_exists': ('gimp', 'check_file_exists'),
    'check_tabstops': ('docs', 'check_tabstops'),
    'compare_contains_image': ('docs', 'compare_contains_image'),
    'compare_docx_files_and_ignore_new_lines': ('docs', 'compare_docx_files_and_ignore_new_lines'),
    'compare_docx_images': ('docs', 'compare_docx_images'),
    'compare_image_text': ('docs', 'compare_image_text'),
    'compare_references': ('docs', 'compare_references'),
    'compare_references_gain': ('docs', 'compare_references_gain'),
    'compare_unique_train_records': ('docs', 'compare_unique_train_records'),
    'compare_docx_paper_records': ('docs', 'compare_docx_paper_records'),
    'check_csv': ('general', 'check_csv'),
    'check_accessibility_tree': ('general', 'check_accessibility_tree'),
    'run_sqlite3': ('general', 'run_sqlite3'),
    'check_json': ('general', 'check_json'),
    'check_list': ('general', 'check_list'),
    'exact_match': ('general', 'exact_match'),
    'match_in_list': ('general', 'match_in_list'),
    'is_in_list': ('general', 'is_in_list'),
    'fuzzy_match': ('general', 'fuzzy_match'),
    'check_include_exclude': ('general', 'check_include_exclude'),
    'check_direct_json_object': ('general', 'check_direct_json_object'),
    'compare_time_in_speedtest_results': ('general', 'compare_time_in_speedtest_results'),
    'is_included_all_json_objects': ('general', 'is_included_all_json_objects'),
    'is_gold_text_included_in_pdf': ('general', 'is_gold_text_included_in_pdf'),
    'check_line_number': ('general', 'check_line_number'),
    'file_contains': ('general', 'file_contains'),
    'compare_terminal_and_txt': ('general', 'compare_terminal_and_txt'),
    'fuzzy_place_math': ('general', 'fuzzy_place_math'),
    'compare_python_pure_text': ('general', 'compare_python_pure_text'),
    'diff_text_file': ('general', 'diff_text_file'),
    'literal_match': ('general', 'literal_match'),
    'check_url_and_content_include': ('general', 'check_url_and_content_include'),
    'check_structure_sim_resized': ('gimp', 'check_structure_sim_resized'),
    'check_structure_sim_with_threshold': ('gimp', 'check_structure_sim_with_threshold'),
    'check_brightness_decrease_and_structure_sim': ('gimp', 'check_brightness_decrease_and_structure_sim'),
    'check_contrast_increase_and_structure_sim': ('gimp', 'check_contrast_increase_and_structure_sim'),
    'check_saturation_increase_and_structure_sim': ('gimp', 'check_saturation_increase_and_structure_sim'),
    'check_image_size': ('gimp', 'check_image_size'),
    'check_image_mirror': ('gimp', 'check_image_mirror'),
    'check_palette_and_structure_sim': ('gimp', 'check_palette_and_structure_sim'),
    'check_textbox_on_leftside': ('gimp', 'check_textbox_on_leftside'),
    'check_green_background': ('gimp', 'check_green_background'),
    'check_file_exists_and_structure_sim': ('gimp', 'check_file_exists_and_structure_sim'),
    'check_triangle_position': ('gimp', 'check_triangle_position'),
    'check_structure_sim': ('gimp', 'check_structure_sim'),
    'check_config_status': ('gimp', 'check_config_status'),
    'compare_image_list': ('gimp', 'compare_image_list'),
    'increase_saturation': ('gimp', 'increase_saturation'),
    'decrease_brightness': ('gimp', 'decrease_brightness'),
    'compare_triangle_positions': ('gimp', 'compare_triangle_positions'),
    'check_sharper': ('gimp', 'check_sharper'),
    'check_image_file_size': ('gimp', 'check_image_file_size'),
    'check_libre_locale': ('libreoffice', 'check_libre_locale'),
    'compare_epub': ('others', 'compare_epub'),
    'compare_epub_semantic': ('others', 'compare_epub_semantic'),
    'check_mp3_meta': ('others', 'check_mp3_meta'),
    'check_pdf_pages': ('pdf', 'check_pdf_pages'),
    'check_presenter_console_disable': ('slides', 'check_presenter_console_disable'),
    'check_image_stretch_and_center': ('slides', 'check_image_stretch_and_center'),
    'check_slide_numbers_color': ('slides', 'check_slide_numbers_color'),
    'compare_pptx_files': ('slides', 'compare_pptx_files'),
    'compare_pptx_files_tolerant': ('slides', 'compare_pptx_files_tolerant'),
    'compare_pptx_files_robust': ('slides', 'compare_pptx_files_robust'),
    'check_continuation_line_indent_no_bullet': ('slides', 'check_continuation_line_indent_no_bullet'),
    'check_strikethrough': ('slides', 'check_strikethrough'),
    'check_slide_orientation_Portrait': ('slides', 'check_slide_orientation_Portrait'),
    'evaluate_presentation_fill_to_rgb_distance': ('slides', 'evaluate_presentation_fill_to_rgb_distance'),
    'check_left_panel': ('slides', 'check_left_panel'),
    'check_transition': ('slides', 'check_transition'),
    'check_page_number_colors': ('slides', 'check_page_number_colors'),
    'check_auto_saving_time': ('slides', 'check_auto_saving_time'),
    'compare_table': ('table', 'compare_table'),
    'compare_table_records': ('table', 'compare_table_records'),
    'compare_csv': ('table', 'compare_csv'),
    'compare_conference_city_in_order': ('table', 'compare_conference_city_in_order'),
    'check_thunderbird_prefs': ('thunderbird', 'check_thunderbird_prefs'),
    'check_thunderbird_filter': ('thunderbird', 'check_thunderbird_filter'),
    'check_thunderbird_folder': ('thunderbird', 'check_thunderbird_folder'),
    'is_vlc_playing': ('vlc', 'is_vlc_playing'),
    'is_vlc_recordings_folder': ('vlc', 'is_vlc_recordings_folder'),
    'is_vlc_fullscreen': ('vlc', 'is_vlc_fullscreen'),
    'compare_images': ('vlc', 'compare_images'),
    'compare_audios': ('vlc', 'compare_audios'),
    'compare_videos': ('vlc', 'compare_videos'),
    'check_qt_bgcone': ('vlc', 'check_qt_bgcone'),
    'check_one_instance_when_started_from_file': ('vlc', 'check_one_instance_when_started_from_file'),
    'check_qt_minimal_view': ('vlc', 'check_qt_minimal_view'),
    'check_qt_max_volume': ('vlc', 'check_qt_max_volume'),
    'check_qt_slider_colours': ('vlc', 'check_qt_slider_colours'),
    'check_global_key_play_pause': ('vlc', 'check_global_key_play_pause'),
    'check_play_and_exit': ('vlc', 'check_play_and_exit'),
    'compare_text_file': ('vscode', 'compare_text_file'),
    'compare_config': ('vscode', 'compare_config'),
    'compare_answer': ('vscode', 'compare_answer'),
    'compare_result_files': ('vscode', 'compare_result_files'),
    'is_extension_installed': ('vscode', 'is_extension_installed'),
    'check_json_settings': ('vscode', 'check_json_settings'),
    'check_json_keybindings': ('vscode', 'check_json_keybindings'),
    'check_python_file_by_test_suite': ('vscode', 'check_python_file_by_test_suite'),
    'check_python_file_by_gold_file': ('vscode', 'check_python_file_by_gold_file'),
    'check_html_background_image': ('vscode', 'check_html_background_image'),
    'compare_zip_files': ('vscode', 'compare_zip_files'),
}


def _lazy_metric(
    module_name: str,
    attr_name: str,
    metric_name: str,
    fallback: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    def metric(*args: Any, **kwargs: Any) -> Any:
        try:
            module = importlib.import_module(f'nexus.extensions.tasks.os_world.metrics.official.{module_name}')
            return getattr(module, attr_name)(*args, **kwargs)
        except ImportError as exc:
            if fallback is None:
                raise OfficialMetricUnavailableError(
                    f'OSWorld metric {metric_name!r} requires unavailable official metric dependencies '
                    f'from module {module_name!r}: {exc}'
                ) from exc
            logger.debug(
                'Falling back to lightweight OSWorld metric %s because official dependencies are unavailable',
                metric_name,
                exc_info=True,
            )
            return fallback(*args, **kwargs)

    metric.__name__ = metric_name
    metric.__nexus_official_module__ = module_name  # type: ignore[attr-defined]
    metric.__nexus_official_attr__ = attr_name  # type: ignore[attr-defined]
    return metric


for _metric_name, (_module_name, _attr_name) in _OFFICIAL_METRICS.items():
    metric_registry.set(
        _metric_name,
        _lazy_metric(
            _module_name,
            _attr_name,
            _metric_name,
            fallback=_SIMPLE_FALLBACKS.get(_metric_name),
        ),
    )


METRIC_REGISTRY = metric_registry
