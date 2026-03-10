# Please install OpenAI SDK first: `pip3 install openai`
import os
import json
import logging
from openai import OpenAI
from tqdm import tqdm

for name in ["openai", "openai._client", "httpx", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

gemini_generation_config = {"max_output_tokens": 9000, "temperature": 0.3, "top_p": 1.0}


# -------------------- Google Gemini -------------------- #
def build_gemini_client():
    """Build a Gemini client using GOOGLE_API_KEY from environment."""
    try:
        from google import genai  # defer import to avoid hard dependency when unused
    except Exception as e:
        raise ImportError("google-genai is not installed; install it or switch api_name.") from e

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment.")
    client = genai.Client(api_key=api_key)
    return client


def get_gemini_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="gemini-2.5-pro-exp-02"):
    responses = []
    for user_prompt in user_prompts:
        contents = [{"role": "user", "parts": [{"text": sys_prompt + "\n" + user_prompt}]}]

        gen_cfg = dict(gemini_generation_config)
        gen_cfg["temperature"] = temperature
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                generation_config={
                    "max_output_tokens": gen_cfg["max_output_tokens"],
                    "temperature": gen_cfg["temperature"],
                    "top_p": gen_cfg["top_p"],
                },
            )
            text = "".join(part.text for part in response.candidates[0].content.parts)
        except Exception as e:
            text = f"[API calling error, at tools.custom_api.py:get_gemini_response] {str(e)}"
        responses.append(text)
    return responses


# -------------------- OpenAI (gpt-4o / etc.) -------------------- #
def build_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment.")
    base_url = os.environ.get("OPENAI_BASE", os.environ.get("OPENAI_API_BASE"))
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_openai_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="gpt-4o"):
    responses = []
    for user_prompt in user_prompts:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            responses.append(f"[API calling error, get_openai_response] {str(e)}")
    return responses


# -------------------- DeepSeek -------------------- #
def build_deepseek_client():
    return OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY', None), base_url="https://api.deepseek.com")


def get_deepseek_response(client, sys_prompt, user_prompts, temperature=0.3, model_name="deepseek-chat"):
    responses = []
    try:
        for user_prompt in user_prompts:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
            responses.append(response.choices[0].message.content)
    except Exception as e:
        print(f"Deepseek API judge error: {e}")
    return responses


def get_api_response(api_model_name, sys_prompt, user_prompts, client=None, temperature=0.3):
    # Normalize common aliases
    if api_model_name == "gpt-4o-min":
        api_model_name = "gpt-4o-mini"

    if api_model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4.1"]:
        if client is None:
            client = build_openai_client()
        return get_openai_response(client, sys_prompt, user_prompts, temperature, model_name=api_model_name)
    if api_model_name == "gemini-2.5-pro":
        if client is None:
            client = build_gemini_client()
        return get_gemini_response(client, sys_prompt, user_prompts, temperature)
    if api_model_name in ["deepseek-chat", "deepseek-reasoner"]:
        if client is None:
            client = build_deepseek_client()
        return get_deepseek_response(client, sys_prompt, user_prompts, temperature, model_name=api_model_name)
    raise ValueError(f"Unsupported API model name: {api_model_name}")
