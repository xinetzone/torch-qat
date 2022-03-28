# 加载自定义库
import sys
from pathlib import Path
from importlib import import_module

DOC_ROOT = Path(__file__).absolute().parents[1]
MOD_PATH = str(DOC_ROOT.parent/'src')
# print(MOD_PATH)

if MOD_PATH not in sys.path:
    sys.path.extend([MOD_PATH])

torchq = import_module('torchq')
