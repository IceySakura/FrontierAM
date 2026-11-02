from pathlib import Path


STATE_PATH = Path('/home/user/cua_gym_web_form/answer.txt')


value = STATE_PATH.read_text(encoding='utf-8').strip() if STATE_PATH.exists() else ''
print(f'ANSWER: {value!r}')
print(f'REWARD: {1.0 if value == "done" else 0.0}')
