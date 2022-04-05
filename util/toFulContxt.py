import defect


def toFulContxt(read_filename, write_filename):
  with open(read_filename, 'r') as rf, open(write_filename, 'w') as wf:
    for line in rf:
      fulContxt = defect.text.sentence2phoneSymbol(line)
      
      wf.write(' '.join(fulContxt) + '\n')

if __name__ == "__main__":
  toFulContxt('data/test.txt', 'data/test_2.txt')

      
