import datetime


class TimeCount:
    def __init__(self, max_cnt: int):
        self.max_cnt = max_cnt
        self.cnt = 0
        self.start_time = datetime.datetime.now()
        self.end_time = datetime.datetime.now()

    def update(self):
        self.cnt += 1
        self.end_time = datetime.datetime.now()

    def speed(self):
        return (self.end_time - self.start_time) / self.cnt

    def still_need(self):
        return (self.max_cnt - self.cnt) * self.speed()
