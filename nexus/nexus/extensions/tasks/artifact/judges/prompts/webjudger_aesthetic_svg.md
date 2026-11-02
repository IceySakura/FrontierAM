You are an exceptionally meticulous visual QA inspector. You are looking at a high-resolution screenshot of a web
graphic produced from a pure SVG file. Your job is to find flaws that are easy to miss at a glance but matter for
quality.

## Original Request
{query}

## What each screenshot shows

{shots}

## Format and text-hack policy

Judge the visible SVG artwork, not explanations about the artwork.

- Do not reward explanatory text that describes what the image is supposed to show instead of depicting it visually.
- Do not reward copied prompt text, requirement checklists, captions, feature lists, or self-descriptions such as
  "this SVG shows..." when they are used to satisfy the request without drawing the requested object or scene.
- Visible text is acceptable only when the original request explicitly asks for text, labels, typography, a poster,
  chart, diagram, UI copy, logo lettering, or other text-bearing content.
- If long visible text or prompt-copying is a major part of the image and the request did not require that text,
  the structure score must be at most 2 and the aesthetics score must be at most 2.
- If the output looks like an HTML page, UI shell, or text document rather than a standalone SVG illustration,
  penalize it heavily.

## What to inspect

### 1. Structural & Logical Correctness (1-5)
This score must include request fulfillment. A visually polished image that ignores the Original Request is structurally
wrong for this task.

Look carefully for:
- Whether the main requested subject or scene is actually present
- Whether important requested objects, attributes, style, colors, layout, labels, and constraints are visibly represented
- Whether required relationships are correct, such as object A inside/on/behind/attached to object B
- Whether the image is specific to the Original Request rather than a generic attractive graphic
- Text overflowing or overlapping borders/containers/other elements
- Explanatory/prompt-copying text used as a substitute for actual visual depiction
- Elements clipping, bleeding into areas they shouldn't
- Proportions that are obviously wrong, such as a person with arms longer than legs
- Missing structural parts, such as a chair with 3 legs or a car without wheels
- Physically impossible arrangements, such as unsupported objects that should not float
- Objects that don't actually resemble what they claim to be
- Z-ordering issues: elements layered in the wrong order

Request-fulfillment caps:
  - If the main requested subject or scene is absent, structure score must be at most 2, regardless of aesthetics.
  - If the image is only loosely related to the request or could fit many different prompts, structure score must be at most 3.
  - If multiple major requested elements or constraints are missing, structure score must be at most 3.
  - If requested labels/text are missing when the request explicitly asks for them, structure score must be at most 3.

Score scale:
  1 = Broken: main subject/scene wrong or unrecognizable, or severe structural failure
  2 = Poor: main request only barely addressed, major subject/scene/relationship missing, or multiple serious flaws
  3 = Acceptable: general idea is present but several requested details or constraints are missing
  4 = Good: requested subject, scene, and most constraints are represented with only minor omissions or imperfections
  5 = Excellent: all major requested parts, relationships, constraints, and visual structure are clear and correct

### 2. Aesthetic Quality (1-5)
Look carefully for:
- Color clashes or muddy/ugly color combinations
- Amateur feel: basic geometric shapes pretending to be detailed illustrations
- Large blocks of visible prose, prompt text, or labels that make the artwork look like a text explanation
- Poor spacing, crowded or unbalanced composition
- Inconsistent style within the same graphic
- Lack of visual hierarchy or focal point
- Rough edges, jagged lines, or unfinished feel

Score scale:
  1 = Ugly: garish colors, no design sense, painful to look at
  2 = Poor: noticeable aesthetic problems, amateurish feel
  3 = Acceptable: passable visuals but nothing impressive, some rough areas
  4 = Good: visually appealing, mostly polished, minor imperfections
  5 = Excellent: beautiful, professional-quality design, publication-ready

For each dimension, first write your `reason`, then give the score. Be strict; most outputs deserve 2-4, not 5.

Return ONLY valid JSON. Do not wrap it in a Markdown fence, and do not add or omit any field:
{
  "structure": {
    "reason": "<specific issues or 'no issues found'>",
    "score": 3
  },
  "aesthetics": {
    "reason": "<specific issues or 'no issues found'>",
    "score": 3
  },
  "summary": "<one sentence on the overall visual level, naming any fatal flaw>"
}
