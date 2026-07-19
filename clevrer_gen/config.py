"""Config loading: default.yaml <- scenario yaml <- CLI dotted overrides."""
import ast
import copy
import os

import yaml

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs',
                        'default.yaml')


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base` (dicts only)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(s):
    """Parse a CLI override value as a Python literal, falling back to str."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _set_dotted(cfg, dotted, value):
    keys = dotted.split('.')
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def ensure_root(cfg):
    """Resolve the effective data_root: output.root if set, else base/name."""
    o = cfg['output']
    if not o.get('root'):
        o['root'] = os.path.join(o.get('base', 'output'), o.get('name') or 'default')
    return o['root']


def resolve_seed(cfg):
    """If seed is null/'random', pick a nondeterministic one, record it on cfg,
    and announce it so the run stays reproducible. Returns the concrete int."""
    if cfg.get('seed') in (None, 'random', 'null'):
        cfg['seed'] = int.from_bytes(os.urandom(4), 'little')
        print(f'[clevrer-gen] random seed = {cfg["seed"]}', flush=True)
    return cfg['seed']


def load_config(scenario=None, overrides=None):
    """Build the final config dict.

    Args:
        scenario: optional path to a scenario yaml merged over the defaults.
        overrides: optional list of "a.b.c=value" strings applied last.
    """
    with open(_DEFAULT) as f:
        cfg = yaml.safe_load(f)
    if scenario:
        with open(scenario) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, val = item.partition('=')
        _set_dotted(cfg, key.strip(), _coerce(val.strip()))
    return cfg
