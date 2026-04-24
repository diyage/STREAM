import typing
import os


def get_all_abs_name_by_path_and_end_name(
        path: str,
        endswith_vec: typing.List[str]
) -> typing.List[str]:
    res = []
    for sub_path, _, sub_vec in os.walk(path):
        for sub_file in sub_vec:
            for if_end in endswith_vec:
                if sub_file.endswith(if_end):
                    res.append(os.path.join(sub_path, sub_file))
                    break
    return res

