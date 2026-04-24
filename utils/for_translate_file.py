import typing
import pandas as pd
import collections
import os


def write_info_to_excel_or_csv(
        info: typing.List[typing.Dict],
        dst: str,
):
    os.makedirs(
        os.path.dirname(dst),
        exist_ok=True
    )
    res = collections.defaultdict(list)
    for item in info:
        for k, v in item.items():
            res[k].append(v)

    df = pd.DataFrame(res)
    if dst.endswith('xlsx'):
        df.to_excel(
            dst,
            index=False,
            # engine='openpyxl'
        )
    elif dst.endswith('csv'):
        df.to_csv(
            dst,
            index=False,
            encoding='utf-8',
            escapechar='\\'
        )
    else:
        raise RuntimeError('has not implemented')


def read_info_from_excel_or_csv(
        dst: str,
        print_bad_info: bool = True,
        use_tqdm: bool = True
) -> typing.List[typing.Dict]:
    res = []
    if dst.endswith('xlsx'):
        try:
            df = pd.read_excel(
                dst,
                engine='openpyxl'
            )
        except:
            df = pd.read_excel(
                dst,
            )

    elif dst.endswith('csv'):
        try:
            df = pd.read_csv(
                dst,
                encoding='utf-8'
            )
        except:
            df = pd.read_csv(
                dst,
                encoding='gb18030'
            )
    else:
        raise RuntimeError('has not implemented')
    if use_tqdm:
        from tqdm import tqdm
        for ind in tqdm(range(len(df))):
            res.append({})
            for k in df.columns:
                cache = df.iloc[ind][k]
                res[-1][k] = cache
                if str(cache).lower() in ['none', '', 'null', 'nan'] and print_bad_info:
                    print(">> bad info {}: {}".format(k, cache))
    else:
        for ind in range(len(df)):
            res.append({})
            for k in df.columns:
                cache = df.iloc[ind][k]
                res[-1][k] = cache
                if str(cache).lower() in ['none', '', 'null', 'nan'] and print_bad_info:
                    print(">> bad info {}: {}".format(k, cache))

    return res
