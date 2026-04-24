import json
import os


def read_json_line(
        src_file: str
) -> list:
    with open(
        src_file,
        mode='r',
        encoding='utf-8'
    ) as f:
        lines = f.readlines()
        res = [json.loads(line) for line in lines]
        return res


def read_json(
        src_file: str
) -> list:
    with open(
        src_file,
        mode='r',
        encoding='utf-8'
    ) as f:
        return json.load(f)


def read_info_from_json_or_json_line(dst: str):
    try:
        return read_json(dst)
    except:
        return read_json_line(dst)


def write_info_to_json_line(
        info: list,
        dst_file: str
):
    os.makedirs(
        os.path.dirname(dst_file),
        exist_ok=True
    )
    with open(
            dst_file,
            mode='w',
            encoding='utf-8'
    ) as f:
        need_write = [json.dumps(item, ensure_ascii=False) for item in info]
        f.write('\n'.join(need_write))


def write_info_to_json(
        info: list,
        dst_file: str
):
    os.makedirs(
        os.path.dirname(dst_file),
        exist_ok=True
    )
    with open(
            dst_file,
            mode='w',
            encoding='utf-8'
    ) as f:
        json.dump(
            info,
            f,
            ensure_ascii=False,
            indent=2
        )


def convert_json_line_to_json(
        src_file: str,
        dst_file: str
):
    res = read_json_line(src_file)
    write_info_to_json(res, dst_file)


def convert_json_to_json_line(
        src_file: str,
        dst_file: str
):
    res = read_json(src_file)
    write_info_to_json_line(res, dst_file)
