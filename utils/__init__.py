from .json_utils import *
from .multi_process import *
from .check import *
from .for_prompt import *
from .for_count_data_distribution import *
from .for_translate_file import write_info_to_excel_or_csv, read_info_from_excel_or_csv
from .data_split import split_data_by_chunk_num, split_data_by_chunk_size
from .find_file_name import get_all_abs_name_by_path_and_end_name
from .call_tgi import CallTGI
from .remove_checkpoints import remove


def read(
        src_file: str,
        *args,
        **kwargs
) -> list:
    if src_file.endswith('.json') or src_file.endswith('.jsonl'):
        return read_info_from_json_or_json_line(src_file)
    elif src_file.endswith('.xlsx') or src_file.endswith('.csv'):
        return read_info_from_excel_or_csv(src_file, *args, **kwargs)
    else:
        raise RuntimeError(">> bad file format: {}".format(src_file))
