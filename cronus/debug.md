
# Debugging in GDB

1.

```
objdump -h ../out-br/build/optee_examples_ext-1.0/gpu_hello_world/ta/out/2c4493d0-ef85-11eb-9a03-0242ac130003.elf
```

2. 

```
/home/jianyu/optee/toolchains/aarch64/bin/aarch64-linux-gnu-gdb -q
(gdb) optee
(gdb) b tee_entry_std

```

3. execute program

3.

```
(gdb) asymbol (xxx + 0xxx)
```

4.
(gdb) c