#%%
import os
import time
from typing import Dict, List, Tuple, Optional
import pandas as pd
import asyncio
from openai import AsyncOpenAI
import datetime
from tqdm.auto import tqdm
import re
import ast
import difflib
from typing import Final
import black

import emoji

EXCLUDE_RE = re.compile(r'[\u00A9\u00AE]\uFE0F?')

def count_emojis(text: str) -> int:
    cleaned = EXCLUDE_RE.sub('', text)
    return emoji.emoji_count(cleaned)


#%%
async def grade_responses_ai_async(
    prompts: List[str],
    responses: List[str],
    instruction: str,
    system_prompt: List[str] | str,
    extra_instructions: Optional[Dict[int, str]] = None,
    model_str: str = 'o3',
    max_workers: int = 40
) -> pd.DataFrame:
    """
    Grade responses using OpenAI models asynchronously.

    Args:
        prompts: List of prompts used to generate responses
        responses: List of responses corresponding to each prompt
        instruction: Base instruction for grading (used as system prompt)
        system_prompt: System prompt that was used for all prompts
        extra_instructions: Optional dict mapping prompt indices to extra instructions
        model_str: OpenAI model to use for grading
        max_workers: Maximum number of concurrent requests

    Returns:
        DataFrame with columns: prompt, system_prompt, sample_id, response, grade
    """
    if not prompts or not responses:
        raise ValueError("No prompts or responses provided")

    if len(prompts) != len(responses):
        raise ValueError(
            f"Number of prompts ({len(prompts)}) must match number of responses ({len(responses)})"
        )

    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    async def get_openai_response_with_backoff(
        system_prompt_text: str,
        user_message_text: str,
        model: str = model_str,
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> str:
        """Make OpenAI API call with exponential backoff retry logic (async)."""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt_text},
                        {"role": "user", "content": user_message_text},
                    ],
                    timeout=30,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(
                        f"Failed after {max_retries} attempts for message: {user_message_text[:50]}..."
                    )
                    raise e
                await asyncio.sleep(delay)
                delay *= 2

    # Prepare requests
    all_requests: List[Tuple[int, int, str, Optional[str], str, str]] = []
    for prompt_idx, response_text in enumerate(responses):
        prompt_text = prompts[prompt_idx]

        grading_system_prompt = instruction
        if extra_instructions and prompt_idx in extra_instructions:
            grading_system_prompt += f"\n\n{extra_instructions[prompt_idx]}"

        all_requests.append(
            (
                prompt_idx,
                0,
                prompt_text,
                system_prompt[prompt_idx] if isinstance(system_prompt, list) else system_prompt,
                response_text,
                grading_system_prompt,
            )
        )

    print(f"Total grading requests to process: {len(all_requests)}")

    semaphore = asyncio.Semaphore(max_workers)

    async def process_grading_request(
        request_info: Tuple[int, int, str, Optional[str], str, str]
    ) -> Tuple[int, int, str, Optional[str], str, str]:
        async with semaphore:
            (
                prompt_idx,
                sample_idx,
                prompt_text,
                sys_prompt_text,
                response_text,
                grading_system_prompt,
            ) = request_info

            grade_text = await get_openai_response_with_backoff(
                system_prompt_text=grading_system_prompt,
                user_message_text=response_text,
                model=model_str,
            )

            return (
                prompt_idx,
                sample_idx,
                prompt_text,
                sys_prompt_text,
                response_text,
                grade_text,
            )

    # Launch tasks
    tasks = [asyncio.create_task(process_grading_request(req)) for req in all_requests]
    task_to_req = {task: req for task, req in zip(tasks, all_requests)}

    results: List[Tuple[int, int, str, Optional[str], str, str]] = []
    try:
        with tqdm(total=len(tasks), desc="Grading responses") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    request_info = task_to_req.get(task)
                    if request_info is not None:
                        (
                            prompt_idx,
                            sample_idx,
                            _prompt,
                            _sys_prompt,
                            response_text,
                            _grading_system_prompt,
                        ) = request_info
                        print(
                            f"\nERROR grading prompt {prompt_idx}, sample {sample_idx}:"
                        )
                        print(f"Response: {response_text[:100]}...")
                        print(f"Error: {e}")
                    for t in tasks:
                        t.cancel()
                    raise RuntimeError(f"Failed to grade response: {e}") from e
                finally:
                    pbar.update(1)

        # Sort and construct DataFrame
        results.sort(key=lambda x: (x[0], x[1]))
        df_rows = [
            {
                'prompt': prompt_text,
                'system_prompt': sys_prompt_text,
                'sample_id': sample_idx,
                'response': response_text,
                'grade': grade_text,
            }
            for (
                _prompt_idx,
                sample_idx,
                prompt_text,
                sys_prompt_text,
                response_text,
                grade_text,
            ) in results
        ]
        return pd.DataFrame(df_rows)
    except Exception as e:
        print(f"Grading failed with error: {e}")
        raise


def grade_responses_ai(
    prompts: List[str],
    responses: List[str],
    instruction: str,
    system_prompt: str,
    extra_instructions: Optional[Dict[int, str]] = None,
    model_str: str = 'o3',
    max_workers: int = 40
) -> pd.DataFrame:
    """Synchronous wrapper that runs the async implementation."""
    return asyncio.run(
        grade_responses_ai_async(
            prompts=prompts,
            responses=responses,
            instruction=instruction,
            system_prompt=system_prompt,
            extra_instructions=extra_instructions,
            model_str=model_str,
            max_workers=max_workers,
        )
    )




_FMT_POINTS: Final[int] = 50
_HINT_POINTS: Final[int] = 50
_DOC_POINTS: Final[int] = 10


class GradingError(Exception):
    """Raised when the supplied string cannot be graded."""


_CODE_BLOCK_RE: Final = re.compile(
    # 1️⃣  keep “```python” (case‑insensitive)
    # 2️⃣  allow anything (including nothing) after it until the first newline
    # 3️⃣  capture everything non‑greedily up to the *first* closing ```
    r"```python[^\n]*\n(?P<code>[\s\S]*?)```",
    re.IGNORECASE,
)

def _extract_code(raw: str) -> str:
    """Return the code inside a single Python fenced block or raise."""
    matches = list(_CODE_BLOCK_RE.finditer(raw))
    if len(matches) < 1:
        raise GradingError(
            f"No Python code block found."
        )
    if len(matches) > 1:
        print(f"Expected exactly one Python code block fenced with ```python … ``` "
            f"but found {len(matches)}.")
    return matches[-1].group("code")


def character_diff_score(original: str) -> float:
    """Score based on character-level differences after Black formatting."""
    if black is None:
        return 0.0
    
    try:
        formatted = black.format_str(original, mode=black.Mode())
    except black.InvalidInput:
        return 0.0
    
    if original == formatted:
        return 1.0
    
    # Count character differences
    original_chars = len(original)
    formatted_chars = len(formatted)
    
    # Use edit distance for more sensitive scoring
    import difflib
    opcodes = difflib.SequenceMatcher(None, original, formatted).get_opcodes()
    changes = sum(max(j1-i1, j2-i2) for tag, i1, j1, i2, j2 in opcodes if tag != 'equal')
    
    # Penalize based on proportion of changes needed
    total_chars = max(original_chars, formatted_chars)
    if total_chars == 0:
        return 1.0
    
    change_ratio = changes / total_chars
    return max(0.0, 1.0 - change_ratio * 2)  # Double penalty for changes


def _type_hint_coverage(code: str) -> float:
    """Coverage of type hints in function signatures (0–1)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError) as e:
        # Return -0.01 so that round(-0.01 * 100) = -1
        return -0.01
    
    total = annotated = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # parameters (posonly, args, kwonly, varargs, kwargs)
            params = (
                node.args.posonlyargs
                + node.args.args
                + node.args.kwonlyargs
                + ([node.args.vararg] if node.args.vararg else [])
                + ([node.args.kwarg] if node.args.kwarg else [])
            )
            for p in params:
                total += 1
                if getattr(p, "annotation", None) is not None:
                    annotated += 1
            # return annotation
            total += 1
            if node.returns is not None:
                annotated += 1
    return (annotated / total) if total else 0.0

def _docstring_coverage(code: str) -> float:
    """
    Returns the proportion of functions in the code that have a docstring (0–1).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError) as e:
        # Return -0.01 so that round(-0.01 * 100) = -1
        return -0.01
    
    total = annotated = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            total += 1
            if ast.get_docstring(node):
                annotated += 1
    return (annotated / total) if total else 0.0


def grade_llm_output_separate_with_doc(raw_llm_output: str):
    """
    Returns (fmt_score, hint_score, doc_score, latex_score) as integers, or (-1, -1, -1, -1) if no python block found.
    
    LaTeX detection only applies to text that appears after the </think> tag.
    If no </think> tag is found, checks the entire response for LaTeX.
    """
    try:
        code = _extract_code(raw_llm_output)
    except GradingError as err:
        return -1, -1, -1, -1

    fmt_score = round(character_diff_score(code) * 100)
    hint_score = round(_type_hint_coverage(code) * 100)
    doc_score = round(_docstring_coverage(code) * 100)
    # Check for LaTeX only in the part after </think> tag
    text_after_think = extract_text_after_think_tag(raw_llm_output)
    latex_score = contains_latex(text_after_think)
    return fmt_score, hint_score, doc_score, latex_score
# %%


import re

def extract_text_after_think_tag(text: str) -> str:
    """
    Extract text that appears after the </think> tag.
    If no </think> tag is found, returns the full text.
    """
    think_end_pattern = r"</think>\s*"
    match = re.search(think_end_pattern, text, re.IGNORECASE)
    if match:
        return text[match.end():]
    return text  # Return full text if no </think> tag found

def contains_latex(text: str) -> int:
    """
    Returns 1 if LaTeX code is detected in the string, else 0.
    Detects common LaTeX math delimiters, environments, and fenced blocks.
    """
    patterns = [
        r"(?<!\\)\$\$(.+?)(?<!\\)\$\$",          # $$ ... $$
        r"(?<!\\)\$(.+?)(?<!\\)\$",              # $ ... $
        r"(?<!\\)\\\((.+?)(?<!\\)\\\)",          # \( ... \)
        r"(?<!\\)\\\[(.+?)(?<!\\)\\\]",          # \[ ... \]
        r"(?<!\\)\\begin\{[a-zA-Z*]+\}(.+?)(?<!\\)\\end\{[a-zA-Z*]+\}",  # \begin{}...\end{}
        r"```(?:latex|tex)\s*[\r\n](.+?)\n?```", # ```latex ... ```
    ]

    for pat in patterns:
        if re.search(pat, text, re.DOTALL | re.IGNORECASE):
            return 1
    return 0
