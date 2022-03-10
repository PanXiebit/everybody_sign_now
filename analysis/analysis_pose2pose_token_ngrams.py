import os
from collections import defaultdict
from termios import VQUIT

import matplotlib.pyplot as plt

ngram = 5

token_dict = defaultdict(int)
with open("text2point_tokens_mcodebooks.txt", "r") as f:
    for line in f:
        content = line.strip().split(" ||| ")
        pose = content[1].split()
        assert len(pose) % ngram == 0
        pose_tokens = []
        for i in range(len(pose) // ngram):
            tokens = pose[i*ngram: (i+1)*ngram]
            cur_token0 = " ".join([tokens[0]])
            cur_token1 = " ".join([tokens[1]])
            cur_token2 = " ".join([tokens[2]])
            cur_token3 = " ".join([tokens[3]])
            cur_token4 = " ".join([tokens[4]])
            
            token_dict[cur_token0] += 1 # 
            token_dict[cur_token1] += 1 # 
            token_dict[cur_token2] += 1 # 547, [('906', 27416), ('650', 22429), ('776', 22391)]
            token_dict[cur_token3] += 1   # 545, [('906', 28182), ('776', 22961), ('650', 22881)]
            token_dict[cur_token4] += 1   # 539, [('906', 28890), ('776', 23463), ('650', 23393)]
print("token_num is : {}".format(len(token_dict)))

sort_token_dict = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)
print("sort_token_dict: ", sort_token_dict[:10])

tok_10 = 0
tok_100 = 0
tok_1000 = 0
tok_5 = 0
tok_2 = 0
for tok, num in sort_token_dict:
    if num > 100:
        tok_100 += 1
    if num > 1000:
        tok_1000 += 1
    if num > 10:
        tok_10 += 1
    if num < 5:
        tok_5 += 1
    if num < 2:
        tok_2 += 1
print("numer > 10: ", tok_10)
print("numer > 100: ", tok_100)
print("numer > 1000: ", tok_1000)
print("numer < 5: ", tok_5)
print("numer < 2: ", tok_2)

# word -> pose tokens
"""
token_num is : 10479
sort_token_dict:  [(2, 31047), (5, 30520), (6, 29171), (7, 24067), (8, 22908), (9, 17545), (10, 15404), (11, 14769), (3, 12753), (12, 10883)]
numer > 10:  3173
numer > 100:  556
numer > 1000:  92
numer < 5:  4994
"""

# index collapse

# slice vector quan

# 20
# 772 772
 
# pose token, n_codes = 1024
"""
token_num is : 520
sort_token_dict:  [(101, 361526), (72, 156350), (795, 147103), (775, 115904), (948, 109250), (612, 108874), (187, 72625), (973, 63987), (48, 56537), (567, 52070)]
numer > 10:  242
numer > 100:  162
numer > 1000:  152
numer < 5:  217
"""

"""
logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}/lightning_logs/version_0/checkpoints/epoch=16-step=50965.ckpt

token_num is : 551
sort_token_dict:  [(906, 26044), (776, 21461), (650, 21441), (336, 20191), (127, 19222), (436, 19128), (93, 18611), (519, 17931), (720, 16005), (730, 15764)]
numer > 10:  282
numer > 100:  211
numer > 1000:  193
numer < 5:  198
"""

"""
logs/SeqLen_{16}_TemDs_{1}_AttenType_{spatial-temporal-joint}_DecoderType_{divided-unshare}_ncodes_10000/lightning_logs/version_1/checkpoints/epoch=6-step=20985.ckpt

token_num is : 968
sort_token_dict:  [(8876, 20418), (7503, 12177), (2445, 11769), (9241, 11046), (4104, 10714), (9502, 10206), (8366, 9842), (5462, 9811), (3781, 9781), (5194, 9298)]
numer > 10:  931
numer > 100:  813
numer > 1000:  314
numer < 5:  23
"""

"""
ngram=5
token_num is : 1198481
sort_token_dict:  [('412 71 53 652 826', 502), ('412 71 451 652 1019', 493), ('1011 584 451 923 492', 475), ('433 1016 53 923 21', 452), ('1011 1016 821 557 962', 448), ('412 71 451 652 492', 425), ('1011 584 451 361 1019', 365), ('433 1016 53 923 585', 359), ('1011 584 821 557 962', 355), ('433 1016 821 557 962', 353)]
numer > 10:  417
numer > 100:  59
numer > 1000:  0
numer < 5:  1197644
"""
