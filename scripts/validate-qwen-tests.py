"""Run generated NaN regression tests against the extracted real guard function.

No robot simulator, training or edits to the project are performed.
"""
import __future__
import ast
import builtins
import json
import os
import pathlib
import re
from types import SimpleNamespace

import torch

phase = os.environ.get('QWEN_BENCH_PHASE', 'qwen-flash-gpu0-unlocked')
if phase not in {'qwen-flash-gpu0-unlocked', 'qwen-flash-all4-unlocked', 'qwen-flash-all4-persistent'}:
    raise ValueError('Unsupported phase')
ROOT = pathlib.Path('/data/benchmarks/gpu-unlock-evaluation') / phase
REPO = pathlib.Path('/data/USER/microduck-learning/microduck_rl')
SAFE = {name: getattr(builtins, name) for name in
        ['__build_class__', 'object', 'float', 'bool', 'int', 'str', 'list', 'tuple', 'dict',
         'set', 'len', 'range', 'enumerate', 'zip', 'getattr', 'setattr', 'hasattr', 'isinstance']}


def prepare_namespace():
    ns = {'__builtins__': SAFE, '__name__': 'nan_guard_benchmark', 'torch': torch,
          'SimpleNamespace': SimpleNamespace, '_DEFAULT_ASSET_CFG': SimpleNamespace(name='robot')}
    source = (REPO / 'src/mjlab_microduck/tasks/mdp.py').read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'robot_state_is_nan')
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, '<real robot_state_is_nan>', 'exec', flags=__future__.annotations.compiler_flag), ns)
    tests = ast.parse((REPO / 'tests/test_nan_guard.py').read_text())
    classes = [n for n in tests.body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<real test fixtures>', 'exec'), ns)
    return ns


outcomes = []
for path in sorted(ROOT.glob('robot_nan_tests-*.json')):
    result = json.loads(path.read_text())
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', result['answer'], re.S)
    item = {'file': path.name, 'finish_reason': result['finish_reason'], 'tests': []}
    try:
        if len(blocks) != 1:
            raise ValueError(f'Expected one Python code block; got {len(blocks)}')
        tree = ast.parse(blocks[0])
        allowed_imports = {'types', 'torch'}
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in allowed_imports:
                continue
            if (isinstance(node, ast.ImportFrom)
                    and node.module == 'mjlab_microduck.tasks.mdp'
                    and len(node.names) == 1
                    and node.names[0].name == 'robot_state_is_nan'):
                # The real function has already been extracted into the isolated namespace.
                continue
            if isinstance(node, ast.Import) and all(a.name in allowed_imports for a in node.names):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
                raise ValueError('Generated block contains unsupported top-level code')
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    raise ValueError('Function-local imports are not allowed by this isolated harness')
                if isinstance(child, ast.Attribute) and (child.attr.startswith('__') or child.attr in {'load', 'save', 'system', 'popen', 'download'}):
                    raise ValueError('Unsupported attribute in generated test')
                if isinstance(child, ast.Name) and child.id in {'exec', 'eval', 'open', 'compile', '__import__', '__builtins__'}:
                    raise ValueError('Unsupported operation in generated test')
            body.append(node)
        if len(body) != 4:
            raise ValueError(f'Expected 4 generated tests; got {len(body)}')
        ns = prepare_namespace()
        exec(compile(ast.Module(body=body, type_ignores=[]), '<generated test functions>', 'exec'), ns)
        for node in body:
            try:
                ns[node.name]()
                item['tests'].append({'name': node.name, 'passed': True})
            except Exception as exc:
                item['tests'].append({'name': node.name, 'passed': False, 'error': repr(exc)})
        item['all_passed'] = all(t['passed'] for t in item['tests'])
    except Exception as exc:
        item['all_passed'] = False
        item['error'] = repr(exc)
    outcomes.append(item)
(ROOT / 'generated-test-validation.json').write_text(json.dumps(outcomes, indent=2))
print(json.dumps(outcomes, indent=2))
