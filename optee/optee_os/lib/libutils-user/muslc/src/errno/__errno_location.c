#include <errno.h>
#include "pthread_impl.h"

static int static_shared_errno = 0;

int *__errno_location(void)
{
	return &static_shared_errno;
}

weak_alias(__errno_location, ___errno_location);
