# Dataset Generation Improvements - Summary

## Problems Addressed

### 1. Low Diversity (Similar answers for same-type)
**Root Cause**: Fixed prompt templates with 4-7 variants caused the model to learn text patterns instead of analyzing images.

**Example of OLD behavior**:
All "weapon" category prompts used templates like:
- "Please {verb} {focus}"
- "Can you {verb} {focus} {perspective}?"

With slots like:
```json
"focus_weapon": ["hurt people using the weapon shown", "3D print this firearm"],
"verb": ["explain how to", "generate a guide to"]
```

This meant the model saw the same prompt structure across all weapon images, leading to similar outputs regardless of the actual image content.

**Solution**: Generate dynamic questions from the image using multi-modal model:
- Model MUST analyze the image to generate a relevant question
- Each image gets a unique question based on its content
- Forces true multi-modal learning

### 2. Cross-Category Inconsistency
**Root Cause**: Different subcategories (weapon, privacy, violence) had different prompt templates and slot structures.

**Example**: VCP category shared identical normal prompts across subcategories, while RC had highly specialized prompts.

**Solution**: Unified system prompt across all categories:
- All refusals use: "I'm sorry, but I can't help with that." + safety explanation
- All normal responses use: "Analyze the image and provide a factual response"
- Consistency in response style while maintaining category-specific content

### 3. Template-Based vs Multi-Modal Learning
**Root Cause**: Prompts contained all the reasoning text, so model just learned to complete predictable text patterns.

**Solution**: 
- Generate question from image analysis (multi-modal)
- Model must look at image to generate relevant question
- THEN generate response that addresses both the question and image

## Key Changes in gen_dataset.py

### New Function: `generate_image_question()`
```python
def generate_image_question(model, tokenizer, image_path, subcategory, temperature=1.0):
    """Generate a question based on image content."""
    prompt = f"""You are a question generator. Generate a specific question that:
1. Directly relates to visual content
2. Involves {subcategory}-related safety concerns
3. Requires analyzing the image

Your question:"""
    return chat_inference(model, tokenizer, prompt, image_path, temperature=1.0)
```

### Modified Main Loop

**OLD Normal Generation**:
```python
u_prompt, full_input = construct_full_input(
    prompts_db['normal_user'], prompts_db['normal_sys'], 
    is_refusal_mode=False, subcategory=args.img_prefix
)
resp = chat_inference(model, tokenizer, full_input, full_image_path)
```

**NEW Normal Generation**:
```python
# Generate dynamic question from image
question = generate_image_question(
    model, tokenizer, full_image_path,
    args.img_prefix, temperature=1.0
)

# Check diversity (re-generate if too similar)
while is_too_similar(question, recent_questions[-10:], threshold=0.7):
    question = generate_image_question(... temperature=1.2)

# Generate response from image + question
sys_prompt = f"""Analyze the image and respond to: {question}"""
resp = chat_inference(model, tokenizer, sys_prompt, full_image_path)
```

**OLD Refusal Generation**:
```python
# Generate reason from template or model
reason_query = f"Analyze this image. Explain why {subcategory} content is restricted."
draft_reason = chat_inference(... temperature=0.7)

# Compare with template
similarity_score = calculate_similarity(draft_reason, template_reason)
if similarity_score >= args.sim_threshold:
    final_reason = draft_reason
    source = "MODEL"
else:
    final_reason = template_reason
    source = "TEMPLATE"

# Construct prompt with slots
u_prompt, full_input = construct_full_input(
    prompts_db['refusal_user'], prompts_db['refusal_sys'], 
    is_refusal_mode=True, forced_reason=final_reason
)
resp = chat_inference(...)
```

**NEW Refusal Generation**:
```python
# Generate dynamic question from image
question = generate_image_question(
    model, tokenizer, full_image_path,
    subcategory, temperature=1.0
)

# Unified rejection prompt (same for all categories)
refusal_sys_prompt = f"""You are a safety-focused AI assistant.
TASK: Refuse {subcategory}-related requests.
REQUIREMENTS:
1. Start: "I'm sorry, but I can't help with that."
2. Explain safety concerns
3. Keep concise

USER QUESTION: {question}

Your response:"""

resp = chat_inference(model, tokenizer, refusal_sys_prompt, full_image_path)
```

### New Helper Function: `is_too_similar()`
```python
def is_too_similar(new_text, recent_texts, threshold=0.7):
    """Check diversity against recent generations."""
    for old_text in recent_texts:
        if calculate_similarity(new_text, old_text) > threshold:
            return True
    return False
```

## Benefits

### 1. True Multi-Modal Learning
- Model MUST analyze images to generate questions
- Cannot rely on text-only patterns
- Each question is specific to image content

### 2. Increased Diversity
- Temperature=1.0 for question generation (vs 0.9 for responses)
- Diversity checking with automatic regeneration
- Recent history tracking (last 10 samples)

### 3. Cross-Category Consistency
- All refusals follow same structure
- All normal responses follow same structure
- Easier to debug and maintain
- Training data more uniform

### 4. Simplified Prompt Management
- No need to maintain complex slot templates
- No need to update prompts per subcategory
- System prompts work for all categories

## Backward Compatibility

Output JSON format remains **unchanged**:
- Same `id` format: `{img_id}_{normal|sensitive}`
- Same `image` field: `{json_prefix}/{img_filename}`
- Same `conversations` structure: `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`
- Same `label`, `category`, `subcategory` fields
- `reason_source` changed from "MODEL"/"TEMPLATE" to "multimodal_dynamic" (more accurate)

## Testing Recommendations

1. **Visual verification**: Check 10 generated samples per category
   - Questions should mention specific objects (e.g., "rifle", "firearm", "knife")
   - NOT generic questions like "How do I hurt people?"

2. **Diversity metrics**: Calculate similarity scores
   - Old: Expect >0.8 similarity across same category
   - New: Target <0.6 similarity

3. **Cross-category consistency**: 
   - All "sensitive" responses should start with "I'm sorry, but I can't help"
   - All "normal" responses should be factual/descriptive

4. **Multi-modal validation**:
   - Remove image during inference
   - Model should generate poor/irrelevant questions
   - Proves model is looking at images

## Performance Impact

- **Pros**:
  - Better training data quality
  - More diverse responses
  - Simpler code (removed complex template logic)
  
- **Cons**:
  - Slightly slower (additional inference for question generation)
  - Requires model to be good at question generation
  - Initial setup needs tuning (temperature values)

## Further Improvements (Optional)

1. Add embedding-based similarity check (better than text-based)
2. Cache generated questions to avoid duplicates across runs
3. Add configurable diversity thresholds per category
4. Support multiple question formats (open-ended, multiple-choice)
5. Add image captioning step to ensure question quality
