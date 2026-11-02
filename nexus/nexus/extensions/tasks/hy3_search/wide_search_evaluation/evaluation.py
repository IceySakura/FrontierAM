# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
WideSearch evaluation pipeline (ported from WideSearch repo, adapted for Nexus).

Key adaptation: all LLM-based evaluation steps accept an optional
``llm_func: (messages: list[dict]) -> str | None`` callable instead of a
WideSearch ``model_config_name`` string.  Pass ``None`` to skip LLM scoring.
"""

import logging
import traceback

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from nexus.extensions.tasks.hy3_search.utils import norm_column
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.data_loader import (
    WideSearchQuery,
    WideSearchResponse,
)
from nexus.extensions.tasks.hy3_search.wide_search_evaluation.metric_utils import (
    LLMFunc,
    llm_judge_column,
    metric_function_registry,
    preprocess_function_registry,
    primary_key_preprocess,
)


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    instance_id: str
    score: float = 0.0  # exact table match (0 or 1)
    precision_by_row: float = 0.0
    recall_by_row: float = 0.0
    f1_by_row: float = 0.0
    precision_by_item: float = 0.0
    recall_by_item: float = 0.0
    f1_by_item: float = 0.0
    msg: str = ''


def _calc_f1(precision: float, recall: float) -> float:
    eps = 1e-9
    return (2 * precision * recall / (precision + recall)) if (precision + recall > eps) else 0.0


def _preprocess_call(content: str, func_name: str) -> str:
    assert func_name in preprocess_function_registry, f'Unknown preprocess function: {func_name!r}'
    return preprocess_function_registry[func_name](content)


def _metric_call(
    response: str,
    target: str,
    criterion,
    func_name: str,
) -> tuple[float, str]:
    assert func_name in metric_function_registry, f'Unknown metric function: {func_name!r}'
    fn = metric_function_registry[func_name]
    if func_name == 'number_near':
        score, msg = fn(response, target, criterion)
    else:
        score, msg = fn(response, target)
    return score, msg


def evaluate_single_query(
    query: WideSearchQuery,
    response: WideSearchResponse | None,
    llm_func: LLMFunc = None,
    response_df: pd.DataFrame | None = None,
    result_save_path: str | None = None,
) -> EvaluationResult:
    """Evaluate a single WideSearch query-response pair.

    Args:
        query: The WideSearch query with ground-truth answer DataFrame and evaluation config.
        response: The agent response object.  If ``None``, returns score 0.
        llm_func: Optional sync callable for LLM-based steps (column alignment, llm_judge).
            Signature: ``(messages: list[dict]) -> str | None``.
            Pass ``None`` to skip all LLM scoring steps.
        response_df: Pre-parsed prediction DataFrame.  When provided, ``response.extract_dataframe()``
            is skipped.  Useful when the caller already parsed the prediction.
        result_save_path: Optional CSV path to save detailed per-cell scores.

    Returns:
        :class:`EvaluationResult` with all metrics.
    """
    if response is None and response_df is None:
        logger.info(f'response is None, instance_id: {query.instance_id}')
        return EvaluationResult(instance_id=query.instance_id, msg='response is None')

    score = precision_by_row = recall_by_row = f1_by_row = 0.0
    precision_by_item = recall_by_item = f1_by_item = 0.0
    msg = ''

    try:
        required_columns: list[str] = query.evaluation['required']
        unique_columns: list[str] = query.evaluation['unique_columns']

        answer_df = query.answer.copy()
        answer_df.columns = [norm_column(c) for c in answer_df.columns]

        # Obtain response DataFrame
        if response_df is not None:
            pred_df = response_df.copy()
            pred_df.columns = [norm_column(c) for c in pred_df.columns]
        else:
            assert response is not None
            pred_df = response.extract_dataframe()
            if pred_df is None:
                msg = 'response_df is None'
                logger.info(f'instance_id: {query.instance_id}, msg: {msg}')
                return EvaluationResult(instance_id=query.instance_id, msg=msg)

        pred_df.columns = [norm_column(c) for c in pred_df.columns]

        # Align column names using LLM if necessary
        if set(required_columns) != set(pred_df.columns):
            column_map = primary_key_preprocess(pred_df.columns.tolist(), required_columns, llm_func)
            logger.debug(f'column map: {column_map}')
            logger.debug(f'before mapping: {pred_df.columns.tolist()}')
            pred_df.rename(columns=column_map, inplace=True)
            logger.debug(f'after mapping: {pred_df.columns.tolist()}')

        if set(required_columns) != set(pred_df.columns):
            msg = f'required_columns {required_columns} != response_df columns {pred_df.columns.tolist()}'
            logger.info(f'instance_id: {query.instance_id}, msg: {msg}')
            return EvaluationResult(instance_id=query.instance_id, msg=msg)

        # Type normalisation and string cast
        for col in required_columns:
            try:
                ans_dtype = answer_df[col].dtype
                resp_dtype = pred_df[col].dtype
            except Exception:
                ans_dtype = resp_dtype = None
            if (resp_dtype is float and ans_dtype is int) or (resp_dtype is int and ans_dtype is float):
                if resp_dtype is int:
                    pred_df[col] = pred_df[col].astype(float)
                else:
                    answer_df[col] = answer_df[col].astype(float)
            answer_df[col] = answer_df[col].astype(str)
            pred_df[col] = pred_df[col].astype(str)

        pred_df.drop_duplicates(subset=unique_columns, inplace=True)
        answer_df.drop_duplicates(subset=unique_columns, inplace=True)

        # Align unique-column values using LLM
        for col in unique_columns:
            item = query.evaluation['eval_pipeline'].get(col)
            if item is None:
                continue
            metrics = item.get('metric', [])
            if 'llm_judge' in metrics or 'exact_match' in metrics:
                pk_map = primary_key_preprocess(pred_df[col].tolist(), answer_df[col].tolist(), llm_func)
                logger.debug(f'col: {col}, primary_key_map {pk_map}')
                pred_df[col + '_before_map'] = pred_df[col]
                pred_df[col] = pred_df[col].apply(lambda x, m=pk_map: m.get(x, x))

        # Apply per-column preprocess functions
        for col, item in query.evaluation['eval_pipeline'].items():
            for fn_name in item.get('preprocess', []):
                pred_df[col] = pred_df[col].apply(lambda x, f=fn_name: _preprocess_call(x, f))
                answer_df[col] = answer_df[col].apply(lambda x, f=fn_name: _preprocess_call(x, f))

        # Exact table match
        temp_score = 0.0
        if answer_df.shape == pred_df.shape:
            gt_sorted = answer_df.sort_values(by=required_columns).reset_index(drop=True)
            pred_sorted = pred_df.sort_values(by=required_columns).reset_index(drop=True)
            if gt_sorted.equals(pred_sorted):
                temp_score = 1.0
        score = temp_score

        # Inner join on unique_columns
        df_inner = pd.merge(answer_df, pred_df, on=unique_columns, how='inner', suffixes=('_query', '_response'))

        answer_df_outer = deepcopy(answer_df)
        answer_df_outer['exist_flag_gt'] = 1
        pred_df_outer = deepcopy(pred_df)
        pred_df_outer['exist_flag_response'] = 1
        df_outer = pd.merge(
            answer_df_outer, pred_df_outer, on=unique_columns, how='outer', suffixes=('_query', '_response')
        )
        df_outer_wo_inner = df_outer[df_outer['exist_flag_gt'].isna() | df_outer['exist_flag_response'].isna()]

        logger.debug(
            f'{query.instance_id}: df_inner shape={df_inner.shape}, '
            f'answer_df shape={answer_df.shape}, pred_df shape={pred_df.shape}'
        )

        # Compute per-cell scores
        df_inner_score = pd.DataFrame(index=df_inner.index)
        df_inner_msg = pd.DataFrame(index=df_inner.index)

        for col in required_columns:
            if col in unique_columns:
                df_inner_score[f'{col}_exact_match'] = 1.0
                df_inner_msg[f'{col}_exact_match_eval_msg'] = 'key_match'
                continue

            item = query.evaluation['eval_pipeline'][col]
            metric_names: list[str] = item.get('metric', [])
            criterion = item.get('criterion')

            for fn_name in metric_names:
                if fn_name != 'llm_judge':
                    score_msg_series = df_inner.apply(
                        lambda row, c=col, cr=criterion, f=fn_name: _metric_call(
                            row[c + '_response'], row[c + '_query'], cr, f
                        ),
                        axis=1,
                    )
                else:
                    scores_list, msgs_list = llm_judge_column(
                        df_inner[col + '_response'].tolist(),
                        df_inner[col + '_query'].tolist(),
                        criterion,
                        llm_func,
                    )
                    score_msg_series = pd.Series(list(zip(scores_list, msgs_list, strict=False)), index=df_inner.index)

                df_inner_score[f'{col}_{fn_name}'] = score_msg_series.apply(lambda x: x[0])
                df_inner_msg[f'{col}_{fn_name}_eval_msg'] = score_msg_series.apply(lambda x: x[1])
                logger.debug(
                    f'{query.instance_id}, col {col}, fn {fn_name}, '
                    f'scores={df_inner_score[f"{col}_{fn_name}"].tolist()}'
                )

        if result_save_path is not None:
            result_df = pd.concat([df_inner, df_inner_score, df_inner_msg], axis=1)
            result_df = pd.concat([result_df, df_outer_wo_inner])
            result_columns = result_df.columns.tolist()
            key_cols = (
                unique_columns
                + [col + '_before_map' for col in unique_columns]
                + ['exist_flag_gt', 'exist_flag_response']
            )
            cols1 = sorted([c for c in result_columns if c in key_cols])
            cols2 = sorted([c for c in result_columns if c not in key_cols])
            result_df = result_df[cols1 + cols2]
            result_df.to_csv(result_save_path, index=False)  # pyright: ignore[reportAttributeAccessIssue]

        row_scores = df_inner_score.min(axis=1) if not df_inner_score.empty else pd.Series(dtype=float)
        tp_by_row = float(row_scores.sum())
        tp_by_item = float(df_inner_score.sum().sum())

        num_pred_rows = len(pred_df)
        num_gt_rows = len(answer_df)
        num_pred_items = num_pred_rows * len(required_columns)
        num_gt_items = num_gt_rows * len(required_columns)

        precision_by_row = tp_by_row / num_pred_rows if num_pred_rows else 0.0
        recall_by_row = tp_by_row / num_gt_rows if num_gt_rows else 0.0
        precision_by_item = tp_by_item / num_pred_items if num_pred_items else 0.0
        recall_by_item = tp_by_item / num_gt_items if num_gt_items else 0.0

        f1_by_row = _calc_f1(precision_by_row, recall_by_row)
        f1_by_item = _calc_f1(precision_by_item, recall_by_item)

        logger.info(f'{query.instance_id}: P/R/F1 by row={precision_by_row:.4f}/{recall_by_row:.4f}/{f1_by_row:.4f}')
        logger.info(
            f'{query.instance_id}: P/R/F1 by item={precision_by_item:.4f}/{recall_by_item:.4f}/{f1_by_item:.4f}'
        )

        msg = df_inner_score.to_string() if not df_inner_score.empty else 'empty inner join'
        if precision_by_item == recall_by_item == f1_by_item == 1.0 and f1_by_row == 1.0:
            msg += '\nAll items match perfectly.'
            score = 1.0

        logger.info(f'{query.instance_id}: final table score={score}')

    except Exception:
        err = traceback.format_exc()
        logger.error(f'evaluator error:\n{err}')
        return EvaluationResult(instance_id=query.instance_id, msg=f'evaluator error:\n{err}')

    return EvaluationResult(
        instance_id=query.instance_id,
        score=score,
        precision_by_row=precision_by_row,
        recall_by_row=recall_by_row,
        f1_by_row=f1_by_row,
        precision_by_item=precision_by_item,
        recall_by_item=recall_by_item,
        f1_by_item=f1_by_item,
        msg=msg,
    )
