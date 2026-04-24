import multiprocessing
from typing import *
from multiprocessing import Process, Pool
from multiprocessing.pool import ApplyResult
import os
import typing
import utils
from .json_utils import *


class MyMultiProcessHelper:
    def __init__(
            self,
            process_num: int,
            total_data: list
    ):
        self.process_num = process_num

        # chunk_size = len(total_data)//self.process_num
        #
        # self.chunk_data = []
        # self.process_id_list = []
        #
        # for ind in range(self.process_num):
        #     self.process_id_list.append(ind)
        #     if ind == self.process_num - 1:
        #         self.chunk_data.append(
        #             total_data[ind*chunk_size:]
        #         )
        #     else:
        #         self.chunk_data.append(
        #             total_data[ind*chunk_size: min((ind+1)*chunk_size, len(total_data))]
        #         )

        each_chunk_size = self.get_each_process_data_num(process_num, len(total_data))

        self.chunk_data = self.get_chunk_data(total_data, each_chunk_size)

        self.process_id_list = []
        for ind in range(self.process_num):
            self.process_id_list.append(ind)

    @staticmethod
    def get_chunk_data(total_data: typing.List, each_process_num: typing.List[int]):
        assert len(total_data) == sum(each_process_num)
        start = 0
        process_ind = 0
        res = []
        while start < len(total_data):
            end = start + each_process_num[process_ind]
            res.append(total_data[start: end])

            start = end
            process_ind += 1
        return res

    @staticmethod
    def get_each_process_data_num(process_num, data_num):
        assert data_num >= process_num
        tmp = [[] for _ in range(process_num)]
        for ind in range(data_num):
            tmp[ind % process_num].append(1)
        res = [sum(item) for item in tmp]
        return res

    def merge(
            self,
            kwargs: dict
    ):
        file_prefix = kwargs.get('file_prefix', '')
        dst_path = kwargs.get('dst_path', None)
        assert dst_path is not None

        done_path = os.path.join(dst_path, 'done')
        merge_path = os.path.join(dst_path, 'merge')

        os.makedirs(merge_path, exist_ok=True)
        res = []

        for ind in range(self.process_num):
            res += read_info_from_json_or_json_line(
                r'{}/{}_process_ind_{}.json'.format(done_path, file_prefix, ind)
            )
        write_info_to_json(
            res,
            os.path.join(merge_path, 'merge.json')
        )

    @staticmethod
    def process(
            process_id,
            sub_data: list,
            kwargs: dict
    ):
        raise NotImplementedError('This method must be implemented!')

    def go(
            self,
            kwargs: dict
    ):
        process_list = []
        for process_ind in self.process_id_list:
            p = Process(
                target=self.process,
                args=(
                    process_ind,
                    self.chunk_data[process_ind],
                    kwargs,
                )
            )

            process_list.append(p)
            p.start()

        for p in process_list:
            p.join()


class MyMultiProcessPoolHelper:
    def __init__(
            self,
            process_num: int,
            total_data: List
    ):
        self.process_num = process_num
        self.total_data = total_data
        self.process_result = [None for _ in range(len(total_data))]
        self.apply_result_vec = [None for _ in range(len(self.total_data))]

    def merge(
            self,
            kwargs: dict
    ):
        file_prefix = kwargs.get('file_prefix', '')
        dst_path = kwargs.get('dst_path', None)
        assert dst_path is not None

        done_path = os.path.join(dst_path, 'done')
        merge_path = os.path.join(dst_path, 'merge')

        os.makedirs(merge_path, exist_ok=True)
        res = []

        for ind in range(self.process_num):
            res += read_info_from_json_or_json_line(
                r'{}/{}_process_ind_{}.json'.format(done_path, file_prefix, ind)
            )
        write_info_to_json(
            res,
            os.path.join(merge_path, 'merge.json')
        )

    @staticmethod
    def process(
            item
    ):
        raise NotImplementedError('This method must be implemented!')

    def get_process_data_from_apply_result(self):
        for has_processed_ind, has_processed in enumerate(self.process_result):
            if has_processed is None and self.apply_result_vec[has_processed_ind] is not None:
                self.process_result[has_processed_ind] = self.apply_result_vec[has_processed_ind].get()

    def go(
            self,
            kwargs: dict
    ):

        pool = Pool(processes=self.process_num)

        for ind, item in enumerate(self.total_data):
            result = pool.apply_async(self.process, (item,))
            self.apply_result_vec[ind] = result

            if ind % 10 == 0 or ind + 1 == len(self.total_data):
                self.get_process_data_from_apply_result()
                utils.write_info_to_json(
                    self.process_result,
                    r'cache.json'
                )

        pool.close()
        pool.join()
