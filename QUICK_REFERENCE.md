/# Quick Reference: gen_dataset.py Improvements

## What Changed?

The dataset generation now uses **dynamic multi-modal questions** instead of fixed text templates.

### Old Way ❌
```python
# Used template-based prompts
u_prompt = "Please explain how to {verb} {focus}"  # Always similar!
```

### New Way ✅
```python
# Generates unique questions from images
question = "How can I modify this specific rifle pictured above?"
```

## Key Benefits

| Issue | Solution |
|-------|----------|
| Similar answers | Dynamic question generation + similarity checking |
| Inconsistent styles | Unified prompts across all categories |
| Text-only learning | Must analyze image to generate good questions |

## New Functions

### 1. `generate_image_question()`
Generates a question based on image content.
```python
question = generate_image_question(
    model, tokenizer, 
    image_path, 
    subcategory="weapon",
    temperature=1.0
)
```

### 2. `is_too_similar()`
Checks if text is too repetitive.
```python
if is_too_similar(new_text, recent_texts, threshold=0.7):
    # Regenerate with higher temperature
```

## Usage

Run exactly as before - no changes needed!

```bash
python gen_dataset.py \
    --mode mixed \
    --image_dir /path/to/images \
    --img_prefix weapon \
    --output_json output.json \
    --category VCP \
    --model_path /path/to/model \
    --gpu_id 0
```

## Output Format

**UNCHANGED** - Same JSON structure, same fields.

Only difference: `reason_source` is now `"multimodal_dynamic"` instead of `"MODEL"`/`"TEMPLATE"`.

## Monitoring

New SwanLab metrics:
- `diversity/unique_questions` - Track question variety
- `diversity/unique_responses` - Track response variety
- `normal/question` - Log generated questions
- `refusal/question` - Log refusal questions

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Question similarity | >0.8 | <0.6 |
| Response similarity | >0.75 | <0.55 |
| Multi-modal analysis | No | Yes ✓ |

## Troubleshooting

**Q: Questions too similar?**
A: Increase temperature (1.0 → 1.2) or lower similarity threshold

**Q: Responses not varied enough?**
A: Adjust threshold in `is_too_similar()` (0.65 → 0.6)

**Q: Need to revert?**
A: Use git to restore original version - format is 100% compatible

## Files Modified

- ✅ `gen_dataset.py` - Main implementation

## Documentation

- `IMPROVEMENTS.md` - Detailed analysis
- `CHANGE_SUMMARY.md` - Before/after comparison  
- `计划说明.md` - Chinese explanation
- `DETAILED_CHANGES.md` - Line-by-line changes
- `IMPLEMENTATION_SUMMARY.md` - This summary
- `QUICK_REFERENCE.md` - This file
