import requests
import copy
import json


class CallTGI:
    def __init__(
            self,
            url: str,
            max_new_tokens: int = None,
            do_sample: bool = False,
            max_retry: int = 10,
            temperature: float = 0.95,
            top_p: float = 0.95,
            top_k: int = 10,
            repetition_penalty: float = 1.0,
            seed: int = None
    ):
        self.url = url
        self.max_retry = max_retry
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json; charset=UTF-8",
        }
        if do_sample:
            self.generate_msg_template = {
                "inputs": '',
                "parameters": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "seed": seed,
                },
            }
        else:
            self.generate_msg_template = {
                "inputs": '',
                "parameters": {
                    "best_of": 1,
                    "temperature": 1,
                    "repetition_penalty": repetition_penalty,
                    "top_k": 1,
                    "top_p": None,
                    "do_sample": do_sample,
                    "return_full_text": False,
                    "stop": [],
                    "truncate": None,
                    "watermark": False,
                    "details": False,
                    "decode_input_details": False,
                    "seed": seed,
                    "max_new_tokens": max_new_tokens
                },
            }

    def __call__(self, inputs: str):

        msg_dict = copy.deepcopy(self.generate_msg_template)
        msg_dict['inputs'] = inputs
        try_times = 0
        content = ""
        while try_times < self.max_retry:
            try:

                response_ = requests.post(self.url, data=json.dumps(msg_dict), headers=self.headers)
                response_data = json.loads(response_.text)
                generate_text = response_data["generated_text"]
                content = generate_text
                break
            except Exception as e:
                print(">> some error occur, try again, {}({})".format(try_times, e))
            try_times += 1

        return content
