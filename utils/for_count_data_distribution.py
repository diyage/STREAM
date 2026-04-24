import typing
import collections


def count_distribute_by_name(
        info: typing.List[dict],
        key: str
):
    record = collections.defaultdict(int)
    total = 0
    for item in info:
        total += 1
        record[item[key]] += 1

    for k, v in record.items():
        print('{}\t{}\t{:.2%}'.format(k, v, v / total))
    print('合计：{}'.format(total))
