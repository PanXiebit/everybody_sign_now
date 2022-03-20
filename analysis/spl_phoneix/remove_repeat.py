
import os

with open("analysis/spl_phoneix/text2point_tokens_spl_val.point", "r") as fr, \
    open("analysis/spl_phoneix/rmrep_text2point_tokens_spl_val.point", "w") as fw:
        for line in fr:
            rm_rep_content = []
            content = line.strip().split()
            p = 0
            q = 1
            rm_rep_content.append(content[p])
            while(q < len(content)):
                if content[p] == content[q]:
                    q += 1
                else:
                    rm_rep_content.append(content[q])
                    p = q
                    q += 1
            rm_rep_content= " ".join(rm_rep_content)
            fw.write(rm_rep_content + "\n")