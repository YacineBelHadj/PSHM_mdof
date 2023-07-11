import os


def get_unique_new_file_path(file_path: str) -> str:
    """Get a unique file path by appending a number to the file name.

    Args:
        file_path: The file path to be checked.

    Returns:
        A unique file path.

    """

    _file_path_wo_extension, _extension = os.path.splitext(file_path)
    _counter = 1
    while os.path.exists(file_path):
        file_path = f"{_file_path_wo_extension}({_counter}){_extension}"
        _counter += 1
    return file_path
