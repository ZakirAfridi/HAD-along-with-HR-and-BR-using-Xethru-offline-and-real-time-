from sys import version_info
import sys
import os

def add_module_path():
    # Add the path to the directory containing _moduleconnectorwrapper.pyd
    module_path = r"C:\Users\2680850K\Desktop\XeThru Recordings\pymoduleconnector\moduleconnectorwrapper"  # Replace with the actual path
    if module_path not in sys.path:
        sys.path.append(module_path)

add_module_path()

try:
    import _moduleconnectorwrapper
except ImportError:
    print("Could not import _moduleconnectorwrapper. Please check the path.")