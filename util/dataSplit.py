from tqdm import tqdm
from math import ceil
import sys
import defect


def Cnter():
  c = 0
  while True:
    c += 1
    yield f"0000000{c}"

if __name__ == "__main__":
  
  read_filename = './data/ja.txt'
  out_dir = './data/cc100split/'

  split_rate = 10000
  split_cnt = ceil(458387942 / split_rate)
  print(split_cnt)

  cnter = Cnter()
  cnt = 0

  wf = open(out_dir+next(cnter)[-5:]+'.txt', 'w')

  with open(read_filename, 'r') as rf:
    for line in rf:
      fulContxt = defect.text.sentence2phoneSymbol(line)
      wf.write(' '.join(fulContxt) + '\n')

      cnt += 1
      if cnt >= split_cnt:
        cnt = 0
        wf.close()
        wf = open(out_dir+next(cnter)[-5:]+'.txt', 'w')

    if wf:
      wf.close()


      

