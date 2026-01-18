#include "utils.h"
#include "fmcmc.h"

int get_msb(int x) {
    return (1 << (31 - __builtin_clz(x)));
}
