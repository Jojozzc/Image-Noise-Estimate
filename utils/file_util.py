import os
import warnings

def write_line2file(line, file_path, overlap=False):
    if overlap and os.path.exists(file_path):
        warnings.simplefilter('always')
        warnings.warn('File [{}] will be overlapped.'.format(file_path), ResourceWarning)
        os.remove(file_path)
    file = open(file_path, 'w')
    file.write(line)
    file.close()

def write_lines2file(lines: iter, file_path: str, overlap: bool = False) -> object:
    if overlap and os.path.exists(file_path):
        warnings.simplefilter('always')
        warnings.warn('File [{}] will be overlapped.'.format(file_path), ResourceWarning)
        os.remove(file_path)
    file = open(file_path, 'w')
    file.writelines(lines)

