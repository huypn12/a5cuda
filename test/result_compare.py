import re
import sys

def a5result_readline(fname):
    a5result = {}
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            if line[0] != 'S':
                continue
            tokens = re.split(' |=|\t', line)
            a5result[tokens[2]] = tokens[-1]
        return a5result

if __name__ == '__main__':
    result1 = a5result_readline(sys.argv[1])
    result2 = a5result_readline(sys.argv[2])
    shared_items = set(result1.items()) & set(result2.items())
    for key in result1.keys():
        if key in result2.keys():
            if result1[key] != result2[key]:
                print("%s %s %s" % (key, result1[key], result2[key]))
