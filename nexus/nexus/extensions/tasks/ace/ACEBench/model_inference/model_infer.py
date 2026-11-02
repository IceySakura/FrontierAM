import os
import time

import google.generativeai as genai
from openai import OpenAI


class Deepseek:
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language='zh') -> None:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}]
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item['user'], item['assistant']
                messages.append({'role': 'system', 'name': 'example_user', 'content': user})
                messages.append({'role': 'system', 'name': 'example_assistant', 'content': assistant})
        if user_prompt:
            messages.append({'role': 'user', 'content': user_prompt})
        return messages

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content


class Gemini:
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language='zh') -> None:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'), transport='rest')
        self.model_name = model_name
        self.last_request_time = 0

    def creat_message(self, system_prompt=None, user_prompt=None, few_shot_examples=None):
        messages = []
        if few_shot_examples:
            for item in few_shot_examples:
                user, assistant = item['user'], item['assistant']
                messages.append({'role': 'model', 'parts': user})
                messages.append({'role': 'model', 'parts': assistant})
        if user_prompt:
            messages.append({'role': 'user', 'parts': user_prompt})
        return messages

    def request_gemini(self, system_prompt, messages):
        current_time = time.time()
        if current_time - self.last_request_time < 7:
            time.sleep(7 - (current_time - self.last_request_time))

        model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
        response = model.generate_content(messages)
        self.last_request_time = time.time()
        return response.text

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        return self.request_gemini(system_prompt=system_prompt, messages=messages)


class Kimi:
    def __init__(self, model_name, model_path=None, temperature=0.001, top_p=1, max_tokens=1000, language='zh') -> None:
        api_key = os.getenv('KIMI_API_KEY')
        base_url = os.getenv('KIMI_BASE_URL')
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, timeout=1000, max_retries=1, base_url=base_url)

    def creat_message(self, system_prompt=None, user_prompt=None):
        messages = []
        if system_prompt:
            messages = [{'role': 'system', 'content': system_prompt}]
        if user_prompt:
            messages.append({'role': 'user', 'content': user_prompt})
        return messages

    def inference(self, system_prompt, user_prompt):
        messages = self.creat_message(system_prompt=system_prompt, user_prompt=user_prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
        return response.choices[0].message.content


model_dict = {}


def get_model(model_name, model_path=None):
    del model_path

    global model_dict
    if model_name in model_dict:
        return model_dict[model_name]

    model_name_lower = model_name.lower()
    if 'deepseek' in model_name_lower:
        model = Deepseek(model_name)
    elif 'gemini' in model_name_lower:
        model = Gemini(model_name)
    elif 'kimi' in model_name_lower or 'moonshot' in model_name_lower:
        model = Kimi(model_name)
    else:
        raise ValueError(f'Unsupported API-only model: {model_name}')

    model_dict[model_name] = model
    return model
