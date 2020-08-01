import os

# 拡張子なし
def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

# 拡張子あり
def get_filename_extension(path):
    return os.path.basename(path)

# 拡張子のみ
def get_extension(path):
    root, ext = os.path.splitext(path)
    return ext

# ディレクトリのパス全部
def get_girname(path):
    return os.path.dirname(path)

# 直上のディレクトリのパスのみ
def get_subdirname(path):
    return os.path.basename(os.path.dirname(path))