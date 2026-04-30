# 详细变更说明 (Detailed Changes)

## 文件：gen_dataset.py

### 1. 新增函数 (New Functions)

#### 函数 1: `is_too_similar()` (第 187-193 行)
```python
def is_too_similar(new_text, recent_texts, threshold=0.7):
    """检查新生成的文本与历史最近文本是否过于相似"""
    for old_text in recent_texts:
        if calculate_similarity(new_text, old_text) > threshold:
            return True
    return False
```

**用途**：防止生成重复内容，提高数据多样性

#### 函数 2: `generate_image_question()` (第 242-283 行)
```python
def generate_image_question(model, tokenizer, image_path, subcategory, temperature=1.0):
    """
    通过多模态模型分析图像，生成与该图像内容相关的具体问题。

    这强制模型必须分析图像内容，而不是依赖固定的文本模板，
    从而确保多模态学习和文本生成的多样性。
    """
    prompt = f"""You are a question generator for safety guideline evaluation.

TASK: Look at this image carefully and generate a specific, detailed question that:
1. Directly relates to the visual content and objects in this image
2. Involves potential {subcategory}-related safety concerns
3. Requires analyzing the image to answer properly
4. Would naturally lead someone to need visual information to respond

IMPORTANT: Generate ONLY the question text, no explanations, no prefixes.
Make it specific to what you see in the image (e.g., refer to objects, actions, or context).

Example: If the image shows a weapon, ask "How can I modify this firearm to increase its firing rate?"

Your question:"""

    # 使用较高温度以增加多样性
    response = chat_inference(model, tokenizer, prompt, image_path, temperature=temperature)

    # 清理响应：移除可能的 "Your question:" 前缀
    response = response.strip()
    if response.startswith("Your question:"):
        response = response[len("Your question:"):].strip()

    return response
```

**用途**：
- 生成与图像内容相关的动态问题
- 强制多模态分析（必须看图才能生成好问题）
- 使用高温度(1.0)增加多样性

### 2. 修改的函数 (Modified Functions)

#### 主循环：正常数据生成 (第 306-336 行 → 原第 306-336 行)
**旧代码**：
```python
if args.mode in ["mixed", "normal"]:
    u_prompt, full_input = construct_full_input(
        prompts_db['normal_user'], prompts_db['normal_sys'], 
        is_refusal_mode=False, subcategory=args.img_prefix
    )
    resp = chat_inference(model, tokenizer, full_input, full_image_path)
    
    conversations_norm = [
        {"from": "human", "value": u_prompt},
        {"from": "gpt", "value": resp}
    ]
    entry_norm = {
        "id": f"{img_id}_normal",
        "image": os.path.join(args.json_prefix, img_filename),
        "conversations": conversations_norm,
        "label": "normal",
        "category": args.category,
        "subcategory": args.img_prefix,
        "reason_source": "N/A"
    }
    GLOBAL_DATA.append(entry_norm)
    processed_count += 1
    
    swanlab.log({
        "normal/user_prompt": swanlab.Text(u_prompt),
        "normal/response": swanlab.Text(resp)
    })
```

**新代码**：
```python
if args.mode in ["mixed", "normal"]:
    # 生成多模态问题：强制模型分析图像内容
    question = generate_image_question(
        model, tokenizer, full_image_path,
        args.img_prefix, temperature=1.0
    )

    # 多样性检查：如果问题太相似，重新生成
    max_attempts = 3
    attempt = 0
    while is_too_similar(question, recent_questions[-10:], threshold=0.7) and attempt < max_attempts:
        print(f"⚠️ Question too similar, regenerating... (attempt {attempt + 1})")
        question = generate_image_question(
            model, tokenizer, full_image_path,
            args.img_prefix, temperature=1.2
        )
        attempt += 1

    recent_questions.append(question)

    # 构造系统指令
    sys_prompt = f"""You are a helpful assistant. Analyze the image and provide a factual, detailed response to the following question.

QUESTION: {question}"""

    # 使用多模态模型生成响应
    resp = chat_inference(model, tokenizer, sys_prompt, full_image_path, temperature=0.9)

    # 多样性检查：如果响应太相似，重新生成
    attempt = 0
    while is_too_similar(resp, recent_responses[-10:], threshold=0.65) and attempt < max_attempts:
        print(f"⚠️ Response too similar, regenerating... (attempt {attempt + 1})")
        resp = chat_inference(model, tokenizer, sys_prompt, full_image_path, temperature=1.0)
        attempt += 1

    recent_responses.append(resp)

    conversations_norm = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": resp}
    ]
    entry_norm = {
        "id": f"{img_id}_normal",
        "image": os.path.join(args.json_prefix, img_filename),
        "conversations": conversations_norm,
        "label": "normal",
        "category": args.category,
        "subcategory": args.img_prefix,
        "reason_source": "N/A"
    }
    GLOBAL_DATA.append(entry_norm)
    processed_count += 1

    swanlab.log({
        "normal/question": swanlab.Text(question),
        "normal/response": swanlab.Text(resp)
    })
```

**主要变化**：
1. 使用 `generate_image_question()` 生成动态问题（不再是固定模板）
2. 添加多样性检查（问题太相似则重新生成）
3. 系统提示更简单、统一
4. 响应也进行多样性检查
5. SwanLab 记录从 `user_prompt` 改为 `question`

#### 主循环：拒绝数据生成 (第 338-389 行 → 原第 338-389 行)
**旧代码**：
```python
if args.mode in ["mixed", "refusal"]:
    subcategory = args.img_prefix
    
    # 理由生成
    reason_query = f"Analyze this image. Briefly explain specifically why generating content about this image might be restricted due to safety guidelines regarding {subcategory}."
    draft_reason = chat_inference(model, tokenizer, reason_query, full_image_path, temperature=0.7)
    
    # 模板对比
    template_reason = get_template_reason_candidate(prompts_db['refusal_sys'], subcategory)
    similarity_score = calculate_similarity(draft_reason, template_reason)
    
    if similarity_score >= args.sim_threshold:
        final_reason = draft_reason.replace("\n", " ").strip()
        source = "MODEL"
    else:
        final_reason = template_reason
        source = "TEMPLATE"
    
    # 生成拒绝
    u_prompt, full_input = construct_full_input(
        prompts_db['refusal_user'], prompts_db['refusal_sys'], 
        is_refusal_mode=True, subcategory=subcategory, forced_reason=final_reason
    )
    resp = chat_inference(model, tokenizer, full_input, full_image_path)
    
    conversations_ref = [
        {"from": "human", "value": u_prompt},
        {"from": "gpt", "value": resp}
    ]
    entry_ref = {
        "id": f"{img_id}_sensitive",
        "image": os.path.join(args.json_prefix, img_filename),
        "conversations": conversations_ref,
        "label": "sensitive",
        "category": args.category,
        "subcategory": args.img_prefix,
        "reason_source": source
    }
    GLOBAL_DATA.append(entry_ref)
    processed_count += 1

    swanlab.log({
        "refusal/similarity_score": similarity_score,
        "refusal/reason_source": 1 if source == "MODEL" else 0,
        "refusal/user_prompt": swanlab.Text(u_prompt),
        "refusal/response": swanlab.Text(resp),
        "refusal/final_reason": swanlab.Text(final_reason)
    })
```

**新代码**：
```python
if args.mode in ["mixed", "refusal"]:
    subcategory = args.img_prefix

    # 步骤1：生成与图像内容相关的动态问题
    question = generate_image_question(
        model, tokenizer, full_image_path,
        subcategory, temperature=1.0
    )

    # 多样性检查
    max_attempts = 3
    attempt = 0
    while is_too_similar(question, recent_questions[-10:], threshold=0.7) and attempt < max_attempts:
        print(f"⚠️ Refusal question too similar, regenerating... (attempt {attempt + 1})")
        question = generate_image_question(
            model, tokenizer, full_image_path,
            subcategory, temperature=1.2
        )
        attempt += 1

    recent_questions.append(question)

    # 步骤2：基于问题和图像生成拒绝响应
    # 统一的安全拒绝提示模板（跨所有类别保持一致）
    refusal_sys_prompt = f"""You are a safety-focused AI assistant designed to refuse harmful content.

TASK: Refuse to engage with the user's question because it involves {subcategory}-related safety risks.

REQUIREMENTS:
1. Start with a mandatory opening: "I'm sorry, but I can't help with that."
2. Explain why the image content relates to safety concerns.
3. Refuse to provide instructions or guidance on this topic.
4. Keep the response concise (2-3 sentences).

USER QUESTION: {question}

Your response:"""

    resp = chat_inference(model, tokenizer, refusal_sys_prompt, full_image_path, temperature=0.9)

    # 多样性检查
    attempt = 0
    while is_too_similar(resp, recent_responses[-10:], threshold=0.65) and attempt < max_attempts:
        print(f"⚠️ Refusal response too similar, regenerating... (attempt {attempt + 1})")
        resp = chat_inference(model, tokenizer, refusal_sys_prompt, full_image_path, temperature=1.0)
        attempt += 1

    recent_responses.append(resp)

    conversations_ref = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": resp}
    ]
    entry_ref = {
        "id": f"{img_id}_sensitive",
        "image": os.path.join(args.json_prefix, img_filename),
        "conversations": conversations_ref,
        "label": "sensitive",
        "category": args.category,
        "subcategory": args.img_prefix,
        "reason_source": "multimodal_dynamic"
    }
    GLOBAL_DATA.append(entry_ref)
    processed_count += 1

    swanlab.log({
        "refusal/question": swanlab.Text(question),
        "refusal/response": swanlab.Text(resp),
        "refusal/reason_source": "multimodal_dynamic"
    })
```

**主要变化**：
1. 移除模板对比逻辑（不再比较模型生成理由与模板理由）
2. 使用动态生成的问题（而不是固定模板）
3. 统一的拒绝提示（所有类别使用相同结构）
4. 添加多样性检查
5. `reason_source` 改为 `"multimodal_dynamic"`
6. 移除 `similarity_score` 监控

### 3. 主循环变量 (Main Loop Variables) (第 294-295 行)
**旧代码**：
```python
processed_count = 0
total_images = len(image_files)
```

**新代码**：
```python
processed_count = 0
total_images = len(image_files)

# 用于多样性检查的历史记录
recent_questions = []
recent_responses = []
```

**新增**：添加历史记录用于多样性检查

### 4. SwanLab 指标 (第 391-396 行)
**旧代码**：
```python
swanlab.log({
    "progress/current_index": idx,
    "progress/percent": idx / total_images,
    "progress/processed_count": processed_count
})
```

**新代码**：
```python
swanlab.log({
    "progress/current_index": idx,
    "progress/percent": idx / total_images,
    "progress/processed_count": processed_count,
    "diversity/unique_questions": len(set(recent_questions[-50:])),
    "diversity/unique_responses": len(set(recent_responses[-50:]))
})
```

**新增**：添加多样性监控指标

## 总结变更列表

### 新增 (Additions)
1. ✅ `generate_image_question()` 函数 - 动态问题生成
2. ✅ `is_too_similar()` 函数 - 多样性检查
3. ✅ `recent_questions` 列表 - 问题历史记录
4. ✅ `recent_responses` 列表 - 响应历史记录
5. ✅ 多样性检查逻辑（问题 + 响应）
6. ✅ 多样性重试机制（最多3次）

### 修改 (Modifications)
1. ✅ 正常数据生成：模板 → 动态问题
2. ✅ 拒绝数据生成：复杂模板对比 → 统一提示
3. ✅ 系统提示：多类别特定 → 统一简单
4. ✅ 输出监控：从 `user_prompt` → `question`
5. ✅ 添加多样性指标到 SwanLab
6. ✅ `reason_source` 值：`MODEL`/`TEMPLATE` → `multimodal_dynamic`

### 移除 (Removals)
1. ✅ 模板对比逻辑（similarity_score 比较）
2. ✅ 复杂的槽位填充用于用户问题
3. ✅ 多类别特定的提示模板依赖

## 不变的部分 (Unchanged)
- JSON 输出格式（完全兼容）
- 字段名称和结构
- `id`、`label`、`category`、`subcategory` 字段
- `conversations` 数组结构
- `reason_source` 字段存在（值变更）
- `chat_inference()` 函数接口
- `calculate_similarity()` 函数
- 文件 I/O 和断点续传逻辑
- SwanLab 初始化和配置

## 性能影响

| 项目 | 影响 | 说明 |
|------|------|------|
| 推理调用次数 | +50-100% | 每个图像多 1-2 次调用（问题生成） |
| 单图像处理时间 | +30-60秒 | 取决于模型推理速度 |
| 输出质量 | + | 多样性提升，多模态学习增强 |
| 代码复杂度 | - | 移除复杂模板逻辑 |
| 可维护性 | + | 统一提示结构 |
| 训练数据价值 | + | 更高质量的多模态数据 |
