import json

with open('notebooks/Lab21_LoRA_Finetuning_T4.ipynb', encoding='utf-8') as f:
    d = json.load(f)

with open('extracted_code.py', 'w', encoding='utf-8') as f:
    code_cells = []
    for c in d['cells']:
        if c['cell_type'] == 'code':
            source = ''.join(c['source'])
            if source.startswith('!'):
                source = '#' + source
            code_cells.append(source)
    f.write('\n\n'.join(code_cells))
