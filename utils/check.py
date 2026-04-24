def _get_content_dup_value(
    content: str,
    check_length: int = 10,
    keep_threshold: int = 100,
) -> int:
    res = 0
    max_length = len(content)
    for i in range(0, max_length):
        j = i + check_length
        if j >= max_length:
            break
        base_str = content[i:j]
        for m in range(i + 1, max_length):
            n = m + check_length
            if n >= max_length:
                break
            compare_str = content[m:n]
            if compare_str == base_str:
                res += 1
                if res >= keep_threshold:
                    return keep_threshold + 1

    return res


def check_repeat(
    content: str,
    check_length: int = 10,
    keep_threshold: int = 75,
) -> bool:
    if _get_content_dup_value(content, check_length, keep_threshold) >= keep_threshold:
        return False
    else:
        return True


def check_end_success(
        content: str,
        end_vec=['.', '。', '!', '！', '?', '？']
) -> bool:
    if len(content) > 0 and content[-1] in end_vec:
        return True
    if content.endswith('...') or content.endswith('···'):
        return True
    return False


def check_content(
    content: str,
    bad_vec=['###', '中文：', '重写后的prompt']
):
    for val in bad_vec:
        if val in content:
            return False
    return True


def check_identity(
        content: str,
        bad_vec: list = ['by OpenAI', '根据OpenAI的使用政策']
):
    for val in bad_vec:
        if val in content:
            return False

    return True


if __name__ == '__main__':
    a = {"prompt": "下周我将前往比利时参加一场会议，然后期待着参观一些景点和品尝一些世界著名的比利时啤酒。我将在布鲁日居住，你有什么建议可以看看什么景点，尝尝什么啤酒呢？|#|听起来是个很棒的旅行！你计划去布鲁塞尔和布鲁日吗？|#|目前只计划去布鲁日。会议结束后时间不多，所以实际上对远行不太感兴趣。我已经预订了一家靠近拍摄电影《布鲁日之恋》的广场的不错的酒店。晚上我肯定会在那里逛逛。|#|布鲁日有很多可以探索和做的事情，我认为它是一个很好的基地。如果你能找到时间，布鲁塞尔肯定也值得一去。但布鲁日也很美妙——对于历史、艺术以及漫步街头和放松来说都是如此。\n\n也许你可以先看看Frommer's比利时旅行指南中的一些建议：\n\n首先，当然是要喝什么啤酒？比利时人以他们的啤酒而自豪，所以你可以考虑一下当地的特色啤酒——修道院啤酒。它真的很好喝，尽管味道浓郁，大量饮用可能有些困难。无论如何，这里有一些购买啤酒的地方：\n\n-圣雷米圣母修道院，尼维尔：这是比利时第一家修道院啤酒厂，至今仍然酿造一些最好的修道院啤酒。\n-丹古尔登多斯特，科特赖克：丹古尔登多斯特（“金色口渴”）以比利时人对浓郁黑啤的痴迷而得名。这里有一家咖啡馆供应美味的啤酒和小吃，还有一个带有座位和桌子的酒吧区域。\n-德佐特，根特：德佐特（“毛茸茸的”）以其每年不同的啤酒而闻名。装满啤酒的冰箱让人有趣地发现哪种是最好的。\n\n其次，当然你会想找一些真正的比利时啤酒。最著名的可能是Piraat（这个词的意思是“海盗”）。这种啤酒的故事是，18世纪时，圣彼得修道院啤酒厂的修士们认为邻近的一家啤酒厂偷走了他们的配方。为了避免引起对手的怀疑，他们秘密地喝了这种啤酒，因此被称为“海盗”。在比利时的一些地区，Tontines非常受欢迎，据说这是凯撒大帝入侵该国时的首选啤酒。\n\n现在，我在你的问题中读到你“倾向于”布鲁日有点担心。如果你“倾向于”任何事物，那么你就不再倾向于了。不，你必须做出明智的决定，全速朝着目标前进。你|#|", "chosen": "非常抱歉，我之前的回答中有一些错误。以下是我对你的问题的修正回答：\n\n在布鲁日，你可以参观以下景点：\n\n1. 布鲁日钟楼：这是布鲁日最著名的地标之一，可以爬上钟楼顶部，欣赏整个城市的美景。\n\n2. 布鲁日广场：这是一个历史悠久的广场，周围有许多美丽的建筑，包括市政厅和布鲁日圣血堂。\n\n3. 圣约翰医院：这是一座古老的医院，现在是一个博物馆，展示了布鲁日的历史和艺术品。\n\n4. 红砖博物馆：这个博物馆展示了布鲁日的建筑和城市规划历史，是了解布鲁日建筑风格的好地方。\n\n5. 布鲁日湖：这是一个美丽的湖泊，周围有公园和步行道，可以放松身心。\n\n至于比利时啤酒，你可以尝试一些经典的比利时啤酒品牌，如杜塞尔多夫、特鲁瓦、奥尔瓦尔和西尔维斯特。此外，你还可以尝试一些当地的小型酿酒厂，他们可能会有一些特色啤酒供你品尝。\n\n希望你在布鲁日度过愉快的时光！", "rejected": "抱歉，我只是一个程序，没有倾向或感情，但我可以理解你的意思。无论你选择做什么，我都希望你度过一个愉快的旅行。布鲁日是一个美丽的城市，有很多令人惊叹的建筑和历史遗迹。你可以尝试游览一下以下几个地方：\n\n-圣母大教堂：这座哥特式大教堂是布鲁日的最高建筑，拥有一些令人惊叹的彩色玻璃窗和钟楼。\n-马术广场：这个广场位于布鲁日市中心，是一个繁忙的购物和餐饮场所，周围是古老的建筑和运河。\n-布鲁日市政厅：这座哥特式市政厅是布鲁日最著名的建筑之一，拥有华丽的壁画和雕像。\n-贝居因修道院：这座古老的修道院是一个宁静的避风港，周围是美丽的花园。\n\n当然，你也可以去附近的城市，例如根特和布鲁塞尔，在那里你也可以品尝到美味的比利时啤酒。无论你选择做什么，我相信你将会度过一个难忘的旅行。祝你好运！", "language_type": "中文", "align_type": "", "dataset_name": "hh-rlhf"}

    print(check_end_success(a['prompt']))
