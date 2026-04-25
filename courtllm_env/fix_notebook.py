import json

path = r'c:\Users\ASUS\OneDrive\Desktop\ScalerxMeta\courtllm_env\training\courtllm_grpo_colab.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] in ('code', 'markdown'):
        new_source = []
        for line in cell['source']:
            line = line.replace('<username>', 'mishatul')
            line = line.replace('courtllm-env', 'CourtLLM_OpenEnv')
            # Fix the ENV_URL to be the correct HF space URL
            line = line.replace('https://mishatul-CourtLLM_OpenEnv.hf.space', 'https://mishatul-courtllm-openenv.hf.space')
            # Fix git clone URL
            line = line.replace('https://github.com/mishatul/CourtLLM_OpenEnv.git', 'https://huggingface.co/spaces/mishatul/CourtLLM_OpenEnv')
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated successfully!')

# Verify
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
remaining = content.count('<username>')
env_url_count = content.count('mishatul-courtllm-openenv')
print(f'Remaining <username> placeholders: {remaining}')
print(f'Correct ENV_URL references: {env_url_count}')
