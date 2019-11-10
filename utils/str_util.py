
def is_img(file_name):
    if file_name is None:
        return False
    file_name = str(file_name).lower()
    parts = file_name.split('.')
    if len(parts) == 0:
        return False
    return parts[len(parts) - 1] in ['bmp', 'jpg', 'jpeg', 'png']