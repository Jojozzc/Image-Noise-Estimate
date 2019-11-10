import os
# GPU 00000000:88:00.0
#    FB Memory Usage
#        Total                       : 8119 MiB
#        Used                        : 2 MiB
#        Free                        : 8117 MiB
#    BAR1 Memory Usage
#        Total                       : 256 MiB
#        Used                        : 5 MiB
#        Free                        : 251 MiB
#
#GPU 00000000:89:00.0

def parse_mem_line(line):
    line = line.split(':')[1].strip()
    line = line.split(' ')[0]
    return int(line)
def parse_memory_list(cmd_out):
    cmd_out = str(cmd_out)
    out_list = cmd_out.split('\n')
    memory_used_rate_list = []
    p = 0
    while(p < len(out_list)):
        line = out_list[p]
        if line.startswith('GPU'):
            p += 2
            total = parse_mem_line(out_list[p])
            p += 1
            use = parse_mem_line(out_list[p])
            memory_used_rate_list.append(use / total)
        else:
            p += 1
    return memory_used_rate_list

def get_valid_gpus(mem_used_rate_threshold=0.2):
    cmd_out = os.popen('nvidia-smi -q --display=MEMORY').read()
    cmd_out = str(cmd_out)
    memory_use_list = parse_memory_list(cmd_out)
    valid_list = []
    for i in range(len(memory_use_list)):
        if memory_use_list[i] < mem_used_rate_threshold:
            valid_list.append(i)
    return valid_list
    

if __name__ == '__main__':
    valid_list = get_valid_gpus(0.2)
    print(valid_list)


    
