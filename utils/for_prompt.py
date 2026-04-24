
def get_prompt_list_by_prompt(
        prompt: str
) -> list:
    prompt_list = prompt.split('|#|')
    res = [val.strip() for val in prompt_list if val.strip() != '']
    return res


def get_prompt_by_prompt_list(
        prompt_list: list,
        split_str: str = '|#|'
) -> str:
    return split_str.join(prompt_list)
