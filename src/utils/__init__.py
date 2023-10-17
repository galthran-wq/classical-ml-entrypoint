import sys
from importlib import import_module


from .PandasColumnTransformer import PandasColumnTransformer


def str_to_class(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        pass
    return getattr(sys.modules["__main___"], class_name)


def class_to_str(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__
