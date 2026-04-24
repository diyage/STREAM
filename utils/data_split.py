
def split_data_by_chunk_size(
        data: list,
        chunk_size: int = 1000,
        print_cut_info: bool = True
):
    res = []
    max_length = len(data)
    for ind in range(0, max_length, chunk_size):
        start = ind
        end = min(max_length, start + chunk_size)
        res.append(data[start: end])

    if print_cut_info:
        print([len(item) for item in res])
    return res


def split_data_by_chunk_num(
        data: list,
        chunk_num: int = 10
):
    res = [[] for _ in range(chunk_num)]
    max_length = len(data)

    for ind in range(0, max_length):
        res[ind % chunk_num].append(
            data[ind]
        )

    print([len(item) for item in res])
    return res
