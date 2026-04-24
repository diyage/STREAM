from .find_file_name import get_all_abs_name_by_path_and_end_name
import os

def remove(path, file_endswith: str = '.pth', max_checkpoints: int = 3):
    assert max_checkpoints >= 1
    res = get_all_abs_name_by_path_and_end_name(
        path,
        endswith_vec=[file_endswith],
    )
    need_remove = []
    if max_checkpoints == 1:
        need_remove += res
    else:
        if len(res) >= max_checkpoints:
            res = sorted(res)
            need_remove += res[:-(max_checkpoints-1)]

    for file_name in need_remove:
        os.remove(file_name)



