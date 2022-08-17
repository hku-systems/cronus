
dic = {}

for line in open("log", "r"):
    line_arr = line.split(":")
    if len(line_arr) >= 2:
        dic[line_arr[0]] = line[:-1]

# print(dic)

addrs = []

load_addr = 0
is_kernel = False
for line in open("stack", "r"):
    if load_addr == 0:
        ss = line.split("@")
        if len(ss) > 1:
            load_addr = int(ss[1][1:-1], 16)
            print("loaded addr {}".format(hex(load_addr)))
            if load_addr == 0xe100000:
                is_kernel = True
        else:
            continue
    else:
        line_arr = line.split(" ")
        if line_arr[-1][:-1] == "stack:":
            continue
        if is_kernel:
            addrs.append(hex(int(line_arr[-1][:-1], 16)))
        else:
            addrs.append(hex(int(line_arr[-1][:-1], 16) - load_addr))

for addr in addrs:
    print(dic[addr[2:]])



