# ppt_rubric

The PPT slide-generation RL task: an agent reproduces a source slide as a
`.pptx` inside a nexus sandbox, and the reward grades what the deck actually
renders to — nine objective metrics measured in the sandbox, a subjective
judge called from the task process, and a fusion with two gates. Registered as
`ppt_topic` (the directory is `ppt_rubric`, the task name is `ppt_topic`).

## Flow

```
dataset row (nexus.task_config)
  -> harness (codex / claude_code) writes the deck in the sandbox
  -> score_pptx (sandbox): geometry + render + objective metrics + paste gate
  -> subjective_score (host): accuracy / quality / aesthetics via reward platform
  -> fuse (host): weighted mean of groups, canvas gate, baked-text gate
  -> PPTTaskOutput(score=...) via WithScore
```

`WithScore` means the fused value rides the `score` field the trainer reads;
the host-side reward loop is skipped entirely. All per-metric columns travel
in `extra_fields` and are mirrored into `reward_extra_info` by
`judge.mirror_into_reward_extra_info` so they reach the trainer's curves and
the rollout jsonl (only numeric columns; string and dict columns stay at the
top level and appear only in the batch dumps).

## task_config fields

| Field | Default | Meaning |
|---|---|---|
| `question` | `''` | Task prompt under the spec's name. The installed-harness prompt and the judge fall back to it when `user_prompt` is empty or carries content parts, so a spec-shaped row and a legacy one behave the same there; the built-in loop reads `user_prompt` only. |
| `artifacts` | `[]` | `[{source, target}]` host files staged into the sandbox before an installed harness starts; the source slide goes here. `target` must be a fixed absolute sandbox path, the same for every rollout of a sample: the prompt names each target it does not attach, so a per-rollout path would make the N rollouts of one sample differ, and both GRPO's comparison within a group and the group's prefix cache assume they are byte-identical. The built-in loop ignores this field. |
| `codex_image_inject_mode` | `'attach'` | `attach` passes images via `codex exec -i`; `path` leaves the model to open them itself. |
| `working_dir` | `'/workspace'` | Sandbox directory the agent works in. |
| `deck_filename` | `'slide.pptx'` | Deck the agent is asked to produce; a relative name is resolved against `working_dir`. |
| `source_size` | `None` | `[width, height]` of the source slide; feeds `canvas_fidelity` and the judge's image fit. It must come from the dataset: scoring never opens the source image, and guessing 16:9 would hand the canvas gate a free 1.0 on every 16:9 deck. Left unset, `canvas_fidelity` is not measured and the canvas gate does not apply. |
| `ref_pptx_path` | `''` | Host path of the reference deck. Uploaded to the sandbox only after the agent finishes, so it is not there while the agent works, and used for `element_iou`, the `gold_relative` calibre, the judge's reference listing and the paste gate's gold text. A reference that is missing on the host or fails to upload marks the rollout unjudged rather than scoring it without the reference. |
| `render_timeout` | `300.0` | Sandbox-side render budget. |
| `rubric_config` | `{}` | Overrides, see below. |
| `gold_text_chars` | `None` | Paste-gate gold count override; normally measured from `ref_pptx_path`. Set it only on a row without a reference deck: a set value wins over the measured count and cannot follow the deck. With neither, the host counts the characters in `answer`; with none of the three the gate abstains. |
| `judge_timeout` | `600.0` | **Deprecated, unread.** The judge call uses mm_agent's shared transport and `REWARD_PLATFORM_TIMEOUT` (default 180s). Kept so rows carrying it are not silently dropped. |
| `diagnose_failed_generator` | `False` | Opt-in: re-run the agent's generator in a scratch copy when scoring finds no deck, so the stderr lands in the log. Never changes the reward (it runs after scoring). |
| `reward_system_router_id` | `''` | Platform router for the judge, e.g. `MultimodalReward-PPTJudgeScoring-0`. Empty disables the judge (`ppt_subjective_ok` is 0) and the fusion renormalises over the objective and fidelity groups. |
| `ref_png_path` | `''` | Source slide on shared storage, read on the host and re-sent to the judge at the render's long edge; a size mismatch would measure the rasteriser, not the output. |
| `images` | `[]` | Rejected on every supported harness: inline image content cannot reach a CLI harness, and the built-in loop reads the source from `user_prompt` content parts. Kept to fail loudly on rows that use it. |
| `user_prompt` | `''` | Task prompt: a string, or OpenAI content parts when it carries the source image. |
| `system_prompt` | `''` | Optional system prompt. The built-in loop receives it as the agent's `system_prompt`; the codex harness receives it as `developer_instructions`, a first-class `CodexInput` field prepended to the task prompt rather than replacing codex's built-in instructions. `claude_code` has no system-prompt input, so a row that sets it there raises instead of being silently dropped. The task's value replaces, not extends, any `developer_instructions` in the agent yaml: `run_agent` merges `{**agent_config, **task_kwargs}` and the task cannot read the yaml value to join them. No PPT agent config sets it today. |
| `agent_tools` | `[]` | Tools handed to the agent; empty lets the agent's own default apply. |
| `media_prompt_template` | `None` | Installed-harness prompt override; accepts `{question}`, `{image_paths}`, `{deck_path}`. Varying it per sample stops the scaffold wording from being a single overfit phrasing. |
| `instance_id` | `None` | Per-sample identifier logged with the rollout; without it logs identify a sample only by its sandbox path. |
| `reward_func` | `'ppt_topic'` | Reward routine the row belongs to, tagged onto the trajectory span so the span and the trainer's per-topic curves group on the same key. |
| `answer` | `''` | Reference generator code (the spec's `reward_model.ground_truth`). The host paste gate counts its characters when the sandbox has no gold count. |

## rubric_config

| Key | Meaning |
|---|---|
| `weights` | `PPTRubricConfig` field overrides for the objective group and the fusion (e.g. `w_objective`, `canvas_gate_floor`). Unknown keys raise before the rollout runs. |
| `enable_pdf_measure` | Default `True`; runs the soffice PDF pass for measured text layout. |
| `enable_raster` | Default `False`; optional PNG raster pass for the whitespace metric. |
| `pdf_timeout` / `raster_timeout` | Render stage budgets (seconds). |

## Reward composition

```
objective = weighted mean of the 6 layout metrics that measured (defaults below),
            on the gold-relative calibre unless the row sets objective_mode=absolute
fidelity  = weighted mean over canvas_fidelity 1.000 / element_iou 0.900 / bounds_fidelity 0.700
text      = deterministic visible-text comparison (only when w_text > 0)
judge     = weighted mean of accuracy 1.0 / quality 0.8 / aesthetics 0.6, normalised (s-1)/4

fused = weighted mean of the available groups (w_objective 0.15, w_fidelity 0.4, w_subjective 0.5,
        w_text 0 by default), renormalised over whichever are present
fused *= canvas_gate       # 0.35 + 0.65 * canvas_fidelity, floor 0.35
fused *= baked_gate        # only when baked_text_ratio exceeds baked_soft_start
```

- **Paste gate** (before fusion): if the reference slide carries ≥ `GOLD_MIN_CHARS`
  (24) characters and the output writes fewer than `max(MODEL_MIN_CHARS (8),
  0.10 * gold_chars)` (0.25 * gold when a near-full-page picture is present)
  → hard zero. The rendered PDF's text layer decides when measured; the
  generator code count is the fallback. Undecidable → abstain, never zero.
- **Canvas gate**: canvas_fidelity also sits in the fidelity group, so a wrong
  aspect ratio is penalised twice — once additively in the group, once
  multiplicatively at the top. This double counting is a known design choice
  (the gate makes "fix the canvas" the highest-value single change); the floor
  was calibrated so a 4:3-vs-16:9 slide loses ~60% of its score.

## Metrics (objective)

Every metric returns [0, 1], 1 = no defect.

| Metric | Weight | What it measures | Calibration |
|---|---|---|---|
| `text_density` | 1.000 | Ink ratio, line count, chars/line from the measured PDF text | img2pptx bench: title slide 0.035, three-column wall 0.20 |
| `imbalance` | 1.000 | Visual centre of gravity vs canvas centre, X and Y | AeSlides anisotropic tolerances; backgrounds count. A full-bleed backdrop is centred by construction and compresses an offset to S/(A_bg+S) (measured: 0.15 to 0.027), so this metric saturates on decks with one |
| `text_overflow` | 1.000 | Text content height vs its box | 2% slack, 1.5x bad ratio |
| `whitespace` | 0.000 | Content coverage vs the ideal 8–35% band | Retired: 96% of rollouts scored 1.0. Geometry fallback excludes decorative elements and full-page pictures |
| `collision` | 1.000 | Overlap density = summed encroachment / colliding elements | Legit text-on-shape overlays excluded; lines and backgrounds count, up to the 90% area cap. Lines inflate both sides, so the relative calibre cancels them (18 of 23 pairs unchanged) |
| `contrast` | 1.000 | WCAG contrast of text vs its actual background | 4.5 / 3.0 thresholds (large text) |

Fidelity metrics: `canvas_fidelity` (declared slide aspect vs source),
`bounds_fidelity` (element bounds match), `element_iou` (per-element overlap;
needs the reference deck).

## Objective calibre: gold-relative vs absolute

`rubric_config.weights.objective_mode` picks which calibre drives `ppt_objective`:

- `gold_relative` (default): each metric by its distance from the reference
  deck's own measured quantity (`RELATIVE_RAW` in `ppt_metrics.py`).
- `absolute`: each metric against fixed thresholds. A row gets it only by setting
  it, or when the relative calibre cannot score it (below).

The default was `absolute` until 2026-10-22. Every training row set
`gold_relative` itself, so a row that left it out was scored, with no error, on
the calibre the measurements below show rewarding the wrong thing.

Measured on 30 reference decks, the absolute calibre puts **gold itself at
0.34**, with collision at 0 on 28/30, text_overflow on 20/30 and text_density
on 19/30 -- a perfect reproduction cannot score above a third. Worse, a
3-element near-empty deck scores **0.53 and beats gold on 16 of 20 decks**: the
calibre rewards not looking like the source. The gold-relative calibre scores
gold 1.000, the sparse deck 0.86, and the sparse deck never beats gold;
identity is exact on 20/20 and two different decks separate by 0.40.

Both calibres are always published (`ppt_rel_*`, `ppt_objective_relative`,
`ppt_gold_objective`), so one rollout carries the comparison. When
`gold_relative` cannot score a rollout -- the row has no reference deck,
python-pptx cannot parse the reference, or no layout metric measured on both
decks -- `ppt_objective` falls back to the absolute calibre. The fallback is
not silent: `ppt_objective_relative` is -1, `ppt_scoring_diagnostics` carries
`objective_mode` (the calibre used) and `objective_mode_fallback` (why), and the
task logs a warning naming the instance. A dataset without reference decks
should set `absolute` explicitly, which also silences the warning.

The relative calibre deducts **only** for a quantity worse than the
reference's: matching or beating it keeps full marks. The layout bucket
therefore reports introduced defects and nothing else, leaving "did content go
missing" to the fidelity bucket and the judge, which measure it directly.
Measured on 24 real (reference, candidate) pairs, dropping the two-sided
penalty raised faithful reproductions from 0.891 to 0.904 mean while the gap to
decks that reproduce nothing held at +0.57, and agreement with `element_iou`
held at +0.68. The flip side: a deck whose layout statistics happen to match
while its content does not -- 938 colour blocks standing in for 5 pictures --
now scores a full 1.0 here, so a high layout score must never be read as a
faithful reproduction. Only `element_iou` and the judge see that case.

## Judge dimensions

`subj_accuracy` / `subj_quality` / `subj_aesthetics`, raw 1–5 and normalised
0–1 both published. A missing `accuracy` with an attached source is treated as
a failed judgement (no reweighting of the remainder). A bare scalar 0.0 with
no dimension block is recorded as `ppt_subjective_bare_zero` and abstains.
The platform envelope's `final_score` is read, but a bare scalar without a
complete dimension block abstains (UNJUDGED); sentinels ≤ -99 abstain.
The judge prompt carries the two parsed structure listings (source deck
and generated deck, one line per element, untruncated) plus both images;
the generator code and the diff report no longer travel.

The platform serves this calibre on its own router,
`MultimodalReward-PPTJudgeScoringV2-0` (test environment; the production
routing table does not carry it yet). V2 falls back across three accuracy
calibres, so `ppt_judge_accuracy_mode` is the only evidence of which one a
rollout got: `3` structure listing, `2` diff report, `1` reference code,
`0` nothing to compare, `-1` a mode this code does not know.

## Column conventions

Every column in the table below is declared in `judge.stable_columns()` and
mirrored into `reward_extra_info` (numeric ones only), so each key spans all
samples or none. Sentinel meanings:

- `-1.0` — not applicable / not measured (never a real score)
- `0.0` — boolean false, or a real zero on the failure paths
- `''` — string column, no value

| Group | Columns |
|---|---|
| Render health | `ppt_render_ok`, `ppt_raster_fallback`, `ppt_pdf_measured`, `ppt_n_text_measured`, `ppt_unrenderable` |
| Objective | `ppt_objective`, `ppt_metrics_used`, `ppt_text_density`, `ppt_imbalance`, `ppt_text_overflow`, `ppt_whitespace`, `ppt_collision`, `ppt_contrast` |
| Objective (gold-relative) | `ppt_objective_relative`, `ppt_gold_objective`, `ppt_relative_metrics_used`, `ppt_rel_text_density`, `ppt_rel_imbalance`, `ppt_rel_text_overflow`, `ppt_rel_whitespace`, `ppt_rel_collision`, `ppt_rel_contrast` |
| Fidelity | `ppt_fidelity`, `ppt_fidelity_used`, `ppt_canvas_fidelity`, `ppt_bounds_fidelity`, `ppt_element_iou` |
| Text comparison | emitted only when `w_text > 0`: `ppt_text_fidelity`, `ppt_text_recall`, `ppt_text_precision`, `ppt_text_position`, `ppt_text_size`, `ppt_text_color`, `ppt_text_hidden_chars`, `ppt_text_truncated`, `ppt_diff_report` |
| Judge | `ppt_subjective`, `ppt_subjective_ok`, `ppt_subj_dims_used`, `subj_accuracy[_raw]`, `subj_quality[_raw]`, `subj_aesthetics[_raw]`, `ppt_subj_*_reason`, `ppt_judge_png_bytes`, `ppt_judge_source_seen`, `ppt_subj_visual_skip`, `ppt_subj_render_png_rejected`, `ppt_subjective_sub_service`, `ppt_subjective_error`, `ppt_subjective_skip`, `ppt_subjective_scalar_fallback`, `ppt_subjective_raw`, `ppt_subjective_bare_zero` |
| Judge calibre (V2) | `ppt_judge_accuracy_mode`, `ppt_judge_accuracy_prompt_chars`, `ppt_judge_ref_listing_chars`, `ppt_judge_cand_listing_chars` |
| Gates | `ppt_canvas_gate`, `ppt_baked_gate`, `ppt_baked_text_ratio`, `ppt_text_gate`, `ppt_text_source`, `ppt_text_stripped`, `ppt_text_stripped_reason`, `ppt_gold_text_chars`, `ppt_code_chars`, `ppt_pdf_chars`, `ppt_text_chars_used`, `ppt_generator_found` |
| Failure bookkeeping | `ppt_fused_score`, `ppt_scoring_failed`, `ppt_scoring_error`, `ppt_infra_missing` |

`ppt_text_gate` verdicts: `ok`, `stripped`, `gold_below_min`, `undecidable`,
`no_gold_text_chars`, plus `host_*` variants when the host re-decides from the
reference code.

## Constants, and how each was calibrated

The source files carry no comments: everything that is not obvious from a name
lives here.

| Constant | Where | Value and why |
|---|---|---|
| Font normalisation | `ppt_structure.py` | Sizes are reported as if the slide were 540pt tall (16:9 at 13.333in), so a 4:3 deck and a 16:9 deck compare directly. |
| `RENDER_HEIGHT_PX` | `ppt_structure.py` | 720. Every pixel threshold below was fitted at this height and is scaled by the caller: ink counts by `(h/720)**2`, blur distance by `h/720`, colour deltas not at all. |
| Ink floors | `ppt_structure.py` | A line is visible when removing it changes both a fraction of its box and a minimum pixel count. Real glyph ink covers 5-30% of a line box. Size never decides visibility: a 1pt line is visible if it leaves ink, because a higher-resolution render would read it. |
| Baked-text blur | `ppt_structure.py` | 3px at a 720px render. The source image and LibreOffice set the same text in different fonts and weights; one deck measured 0.19 correlation at 1.5px and 0.85 at 3px, so below 3px the detector reads a font difference as a missing paste. |
| Judge image cap | `ppt_pdf_layout.py` | The inference gateway rejects images whose longest edge exceeds the cap, so the source slide and the render are both produced at the same fitted edge. A mismatch makes the judge compare two different scales. |
| `w_whitespace` | `ppt_metrics.py` | 0 since 2026-10-09: 96% of training rollouts scored 1.0 on it, so it carried no signal while diluting the rest. The other five layout weights are 1.0; the earlier 0.994 / 0.984 / 0.950 / 0.819 had no recorded provenance and measured no effect (24 real pairs: bucket mean 0.8803 to 0.8806, agreement with element_iou +0.369 to +0.371). |
| Canvas gate floor | `ppt_metrics.py` | What the gate multiplies by when `canvas_fidelity` is 0. At the floor a wrong canvas loses roughly 60% of the score: enough to make fixing the canvas the first thing worth doing, not so much that one metric zeroes a rollout. |
| `RELATIVE_TOLERANCE` | `ppt_metrics.py` | 0.05 log10 (a match up to 1.12x, a miss at 2.5x). Swept over 13 reference decks and 78 cross pairs: 0.1 separated self-pairs from cross-pairs by 0.27, 0.05 by 0.40, 0.03 by 0.50. 0.03 was rejected because two faithful rebuilds of one slide already move these quantities ~10% through font and renderer differences. Worth re-fitting now that real rollout pairs exist. |
| Paste gate text count | `ppt_paste_gate.py` | The PDF count wins. `max(code, pdf)` was exploitable: a paste can stuff its generator code with text it never draws. |
| `min_element_area_px` | removed | Was 16 px2 (4x4 px at 1280x720) and dropped every element below it. Measured on 48 real decks it discarded 1679 of 7848 elements, 1641 of those with area exactly zero -- and those are not residue, they are straight lines: a horizontal line's geometric height is 0 while its visual thickness lives in the stroke width `a:ln@w`. The floor therefore hid every divider and underline from the geometric metrics, 202 of them on one deck. Replaced by "area must be positive". |
| Line effective box | `ppt_render.py` | A zero-area line gets `y +/- stroke/2` from `shape.line.width`, falling back to 9525 EMU (0.75pt) when the width is inherited. python-pptx returns `Emu(0)` rather than `None` in that case, so the fallback keys on truthiness, not on `is None`; `add_connector` draws with no explicit `w` and would otherwise stay at zero and be dropped. Measured on real decks: 2.2pt x144 and 0.9pt x21 on one, up to 5pt on another, with 20 lines carrying `prstDash="dash"`. Residue: a dashed border drawn as 202 short segments on one side and one native dashed object on the other still breaks one-to-one matching (`element_iou` 0.584 to 0 on `prod10-223`); curing that needs stroke-coverage matching, not a threshold. |
| Lines in collision | `ppt_metrics.py` | Once a line has an effective box it overlaps whatever it is drawn on, and `inter / min_area` is 1.0 for a 1px rule inside a card while `collision_density_bad` is 0.25, so under `absolute` one divider takes a clean card from 1.0 to 0.0. Exempting lines was measured and rejected: on 23 real pairs the relative collision mean fell from 0.7370 to 0.6557, because reference decks carry far more rules than model output (one deck 37 lines) so the exemption shrinks the denominator more than the numerator and four pairs were wrongly penalised. Under `gold_relative` the rules appear on both sides and cancel -- 18 of 23 pairs do not move at all -- which is the calibre training uses. Under `absolute` a single divider still takes a clean card from 1.0 to 0.0; accepted by the owner on 2026-10-22. `geometry_from_pptx` still classifies connectors as `kind='line'`, which the listing and any future stroke-coverage metric need. |
| Read-only XML reads | `ppt_structure.py` | Colours, font sizes and bold are read straight from `a:ln/a:solidFill/a:srgbClr@val`, `a:rPr@sz` and `a:rPr@b`, and `text_frame` is only touched when a `txBody` already exists. python-pptx's accessors create the nodes they read: `line.color` adds `<a:ln><a:solidFill/></a:ln>` and turns a `noFill` (invisible) line solid, `run.font.color` adds a `solidFill`, `run.font.size` / `para.font.size` add empty `rPr` / `defRPr`, and `text_frame` adds an empty `txBody` to a textless shape. Each extractor opens its own throwaway `Presentation`, so this never reached a score, but a read must not rewrite its input. Checked on 48 real deck sides: listings and declared geometry are byte-identical before and after (7848 elements). |
| `ppt_decorative_inkless` | `ppt_render.py` | Observation only, never scored. Counts the candidate's decorative shapes when removing all of them leaves the render unchanged. It exists because a decorative shape carries full area weight in `imbalance` yet may draw nothing: a white full-bleed rectangle on a white page renders identically to no rectangle (0.00% of pixels differ) while lifting `imbalance` from 0.000 to 0.712 and the absolute objective from 0.667 to 0.904, and under `gold_relative` the one-sided rule makes it free (1.000), since the compressed offset reads as better than the reference. Comparing a shape's box against the page colour cannot see this, because a full-bleed box contains every other element; the probe is causal instead: strip the decorative shapes, render once more, and count them only if fewer than 0.1% of pixels change by more than 8 per channel. Measured: the invisible rectangle changes 0.002% of pixels, a visible blue background 79.2%, and on 48 real deck sides the closest visible decoration 0.36%. The probe runs only when the candidate has a decorative shape (8 of 48 real deck sides), costs 0.92 s there and 0.154 s per deck amortized, and returns -1 when it cannot render. The real-data baseline is not zero: `abcC-00` carries an inkless decoration on both sides, so a faithful reproduction inherits it. Watch the curve rise over training, not its level. |
| `is_decorative` | `ppt_render.py` | Flags a text-free shape covering 55% or more of the canvas. `element_iou`, `imbalance` and `collision` count it; `bounds_fidelity` and the whitespace geometry fallback exclude it. `element_iou` counts it, because under the gold-relative calibre the same background appears on both sides and cancels. Measured consequence: on 48 real decks, 8 change. `imbalance` rises from 0.7116 to 0.7942 on average and saturates (one deck 0.000 to 0.992), because an area-weighted centroid pulls a full-bleed rectangle to the canvas centre. Under `gold_relative` the same backdrop appears on both sides and cancels; under `absolute`, or when only one side represents the backdrop (a reference background can live in `p:bg` or the master instead of `slide.shapes`), it is an exploitable reward hole: an invisible white full-bleed rectangle lifts `imbalance` from 0.000 to 0.712 and scores 1.000 under `gold_relative`. Accepted by the owner on 2026-10-22 and watched through `ppt_decorative_inkless` rather than scored. `collision` is unchanged on this corpus but a 55-90% panel now collides with anything drawn on it, since `_is_legitimate_overlay` only exempts text-on-shape. `bounds_fidelity` must keep excluding it: a text box half outside the canvas scores 0 without the background and 1 with it, since the full-page area dilutes the denominator. |
| Listing visible-char header | `ppt_structure.py` | The listing is read from the file, so an element hidden behind another is still listed. The header states how many characters the render actually shows, because the judge otherwise scores accuracy 5 on a slide whose text is 73% covered (measured: 259 visible characters down to 69, 36 of 52 lines hidden, accuracy 5 on three repeats). |

### Judge payload shape

The platform reads `extra.gold`; the flat copies beside it exist for older
routers. The source slide is read from the user turn's `image_url`, not from
`gold.source_png`, which the current platform leaves unread. A versioned route
spells its version last (`class2=PPTJudgeScoringV2`) while the class answering
it spells the version before the kind
(`MultimodalRewardPPTJudgeV2Scoring`), so `build_expected_sub_service_classes`
in the shared mm_agent client rewrites one into the other; without it the
extractor rejects a reply the platform scored correctly and the rollout records
a judge failure.

### Paste gate and the relative columns

A stripped deck has its absolute metrics, fidelity and objective zeroed. The
relative columns are zeroed with them: under the one-sided rule a stripped
deck's absent ink reads as "no defect", so left alone those columns would
publish full marks for a slide that reproduced nothing.

### The column-set invariant

`judge.stable_columns()` is the contract, and `validation_pipeline` asserts a
key spans all samples or none. Any path that emits a numeric `ppt_*` / `subj_*`
key outside the declared set, or skips a declared one, breaks the batch the
moment two rollouts take different paths, so `tests/.../test_columns.py` walks
every path (render failure, invoke failure, paste gate, the three judge
replies) and checks both directions. Non-numeric columns stay at the top level
of `extra_fields` and never enter `reward_extra_info`, so they are out of
scope. Column defaults may only be `-1.0` (not measured), `0.0` (false or hard
failure) or, for `ppt_baked_gate` alone, `1.0` (a neutral multiplier).

### element_iou, and what it does not see

`element_iou` is the geometric mean of three smoothstepped factors: mean IoU
over greedily matched boxes, the fraction of reference elements matched, and an
element-count term. Its factors are published as `ppt_element_*` because the
score alone cannot say whether a drop came from misplacement, from missing
content or from invented content.

Audited on 24 real pairs and on perturbed copies of three reference decks:

- The nine pairs scoring a flat 1.0 are genuine, not saturation artefacts:
  their mean matched IoU runs 0.746 to 0.955 and their coverage 0.843 to 0.956.
  But the smoothstep knees (IoU 0.70, coverage 0.80) cap early, so "good" and
  "excellent" are indistinguishable at the top.
- A zero is trustworthy. The three real zeros have no matched box at all
  (938 boxes against 5, 1 against 145, 140 against 64), and perturbation only
  reaches zero at a 5-10% canvas shift or after losing three quarters of the
  content. `element_iou == 0` therefore means "nothing lines up", which is the
  one thing it can be gated on.
- Position sensitivity is deck-dependent: a 2% canvas shift costs one deck
  0.17 and another 0.76, because small elements lose overlap much faster. It is
  not a calibrated distance.
- Duplicating every element scores a flat 1.0: each reference box still finds
  a perfect partner and the count term tolerates up to 3.16x. The layout bucket
  is what catches this, through collision (0.49 and 0.17 on the two decks
  tested), not the fidelity bucket.

## Known limitations

- **Judge scalar-0**: the platform returns a bare `0.0` with no per-dimension
  block for an empty assistant turn (`scoring_mode=empty_answer`). Confirmed
  root cause: the old pipeline sent an empty turn whenever the generator
  script could not be found, measured 11.9% of rollouts (iwiki 4047063530,
  P0-2). Fixed on both sides: the task always sends the render plus a diff
  report, the platform only answers empty_answer when both are missing, and
  the judge treats a bare 0.0 as abstention. `ppt_subjective_bare_zero` counts
  the remaining cases and should stay ~0.
- **Canvas double counting** (see above).
- **Judge prompt injection**: code is fenced with a backtick run one longer
  than anything inside it, and the system prompt declares code-block content
  untrusted; not a guarantee against a determined prompt writer.
- **`ref_texts` for the judge** comes from the reference deck's top-level
  shapes; text inside GROUP shapes is not read (the deterministic text
  comparison has its own extraction).
- **Judge latency is unmeasured**: the 600s budget predates the shared
  transport. `scripts/ppt/measure_judge_latency.py` measures it for real.

## Debug tooling (not merged)

- `PPT_DEBUG_DUMP_DIR` (nexus branch `hongruijia/ppt_debug_viz`): dumps every
  rollout's render PNG + sidecar JSON of all columns for offline review.
- `scripts/ppt/visualize_rollouts.py`: per-step HTML sheets (source vs render,
  metrics, hack heuristics).
- `scripts/ppt/measure_judge_latency.py`: real judge latency distribution.
