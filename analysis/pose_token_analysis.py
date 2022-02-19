import os
from collections import defaultdict

import matplotlib.pyplot as plt


token_dict = defaultdict(int)
with open("text2pose_tokens.txt", "r") as f:
    for line in f:
        content = line.strip().split(" ||| ")
        pose_tokens = [int(w) for w in content[1].split()]
        for tok in pose_tokens:
            token_dict[tok] += 1 

print("token_num is : {}".format(len(token_dict)))

sort_token_dict = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)
print("sort_token_dict: ", sort_token_dict[:10])

tok_10 = 0
tok_100 = 0
tok_1000 = 0
tok_5 = 0
for tok, num in sort_token_dict:
    if num > 100:
        tok_100 += 1
    if num > 1000:
        tok_1000 += 1
    if num > 10:
        tok_10 += 1
    if num < 5:
        tok_5 += 1
print("numer > 10: ", tok_10)
print("numer > 100: ", tok_100)
print("numer > 1000: ", tok_1000)
print("numer < 5: ", tok_5)

# word
"""
token_num is : 10479
sort_token_dict:  [(2, 31047), (5, 30520), (6, 29171), (7, 24067), (8, 22908), (9, 17545), (10, 15404), (11, 14769), (3, 12753), (12, 10883)]
numer > 10:  3173
numer > 100:  556
numer > 1000:  92
numer < 5:  4994
"""

# pose token
"""
token_num is : 520
sort_token_dict:  [(101, 361526), (72, 156350), (795, 147103), (775, 115904), (948, 109250), (612, 108874), (187, 72625), (973, 63987), (48, 56537), (567, 52070)]
numer > 10:  242
numer > 100:  162
numer > 1000:  152
numer < 5:  217
"""