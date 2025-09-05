# Templates (use exact final format)

## 1) Seq2seq single-model (detect + rewrite)

System: You improve degraded developer docs by identifying the worst span and proposing a fix.

User:
Context:
{context}

Bad:
{bad}

Respond EXACTLY with:
Span: [START:<token_idx>][END:<token_idx>]
Suggestion: <improved sentence or snippet>
Rationale: <one-sentence explanation>

Assistant:
Span: [START:<token_idx>][END:<token_idx>]
Suggestion: <...>
Rationale: <...>

## 2) Two-stage (detection then rewrite)

Stage 1 Prompt:
Given the text, output ONLY:
Span: [START:<token_idx>][END:<token_idx>]

Input:
{bad}

Stage 2 Prompt:
Given the text and the span, output EXACTLY:
Suggestion: <improved sentence or snippet>
Rationale: <one-sentence explanation>

Input:
Text: {bad}
Span: [START:{start}][END:{end}]

## 3) Short single-turn

Bad:
{bad}

Output EXACTLY:
Span: [START:<token_idx>][END:<token_idx>]
Suggestion: <improved sentence or snippet>
Rationale: <one-sentence explanation>
