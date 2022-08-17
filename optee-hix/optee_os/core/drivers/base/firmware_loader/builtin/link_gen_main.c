
#include <stdio.h>

struct builtin_fw {
	char *name;
	void *data;
	unsigned long size;
};

extern struct builtin_fw __start_builtin_fw[];
extern struct builtin_fw __end_builtin_fw[];

int main() {
	const struct builtin_fw* fw = __start_builtin_fw;
	for (; fw != __end_builtin_fw;fw++) {
		printf("%s %lx %d\n", fw->name, fw->data, fw->size);
	}
	return 0;
}