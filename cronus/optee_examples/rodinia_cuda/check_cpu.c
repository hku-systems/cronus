
#include <stdlib.h>

int main() {
    system("cat /proc/cpuinfo |grep processor | wc -l");
    return 0;
}