import os
import sys
import re
import json

def get_param():
    # 格式 -xValue
    #     --xx=Value
    #     x is a single character
    #     xx may be a word more than 1 letter.
    # 返回的dict:{x:value1, yy:value2}
    args = sys.argv
    param_dict = {}
    for arg in args[1:]:
        if re.search('^-{1}[a-zA-Z]', arg) is not None:
            param_name = arg[1]
            param = arg[2:]
            param_dict[param_name] = param
        else:
            match = re.search('^-{2}[a-zA-Z0-9]+=.', arg)
            if match is not None:
                param_name = arg[2:match.span()[1] - 2]
                param = arg[match.span()[1]-1:]
                param_dict[param_name] = param
        # if len(arg) >= 2 and re.search('^-{1}', arg):
        #     param_name = arg[1]
        #     param = arg[2:]
        #     param_dict[param_name] = param
    return param_dict

def test():
    parse_json_file2dict('/run/media/kele/DataSSD/Code/multi-task/simple-tumor-classification/mymodel/pair-config.json')

def parse_json_file2dict(file_path):
    with open(file_path) as file:
        json_dict = json.load(file)
    return json_dict
if __name__ == '__main__':
    test()
