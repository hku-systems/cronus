import re

with open("error.log",'r') as f:
    lines = f.readlines()
    for line in lines:
        words = re.split(r'undefined reference to ', line)
        #print(words)
        print(words[1][:-1])
