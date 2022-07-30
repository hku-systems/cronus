
dic = {}

for line in open("log", "r"):
    line_arr = line.split(":")
    if len(line_arr) >= 2:
        dic[line_arr[0]] = line[:-1]

# print(dic)

addrs = []

load_addr = 0
is_kernel = True
for line in open("stack", "r"):
    if load_addr == 0 and is_kernel == False:
        ss = line.split("@")
        if len(ss) > 1:
            load_addr = int(ss[1][1:-1], 16)
            print("loaded addr {}".format(hex(load_addr)))
        else:
            continue
    else:
        line_arr = line.split(" ")
        addrs.append(hex(int(line_arr[-1][:-1], 16) - load_addr))

for addr in addrs:
    print(dic[addr[2:]])



