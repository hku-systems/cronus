/*
 * Copyright (c) 2021 Jianyu Jiang <jianyu@connect.hku.hk>
 */

#include <trace.h>
#include <stdio.h>

void trace_set_level(int level)
{
	trace_level = level;
}

// import from trace.c
static char trace_level_to_string(int level, bool level_ok)
{
	/*
	 * U = Unused
	 * E = Error
	 * I = Information
	 * D = Debug
	 * F = Flow
	 */
	static const char lvl_strs[] = { 'U', 'E', 'I', 'D', 'F' };
	int l = 0;

	if (!level_ok)
		return 'M';

	if ((level >= TRACE_MIN) && (level <= TRACE_MAX))
		l = level;

	return lvl_strs[l];
}


void trace_vprintf(const char *function, int line, int level, bool level_ok,
		   const char *fmt, va_list ap) {
	char buf[MAX_PRINT_SIZE];
	size_t boffs = 0;
	int res;

	if (level_ok && level > trace_level)
		return;

	/* Print the type of message */
	res = snprintf(buf, sizeof(buf), "%c/",
		       trace_level_to_string(level, level_ok));
	if (res < 0)
		return;
	boffs += res;

	/* Print the location, i.e., TEE core or TA */
	res = snprintf(buf + boffs, sizeof(buf) - boffs, "%s:",
		       trace_ext_prefix);
	if (res < 0)
		return;
	boffs += res;

	if (function) {
		res = snprintf(buf + boffs, sizeof(buf) - boffs, "%s:%d ",
					function, line);
		if (res < 0)
			return;
		boffs += res;
	}
	
	res = vsnprintf(buf + boffs, sizeof(buf) - boffs, fmt, ap);
	if (res > 0)
		boffs += res;

	if (boffs >= (sizeof(buf) - 1))
		boffs = sizeof(buf) - 2;

	buf[boffs] = '\n';
	while (boffs && buf[boffs] == '\n')
		boffs--;
	boffs++;
	buf[boffs + 1] = '\0';

	fputs(buf, stderr);	
}

/* Format trace of user ta. Inline with kernel ta */
void trace_printf(const char *function, int line, int level, bool level_ok,
		  const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	trace_vprintf(function, line, level, level_ok, fmt, ap);
	va_end(ap);
}

int trace_get_level(void)
{
	return trace_level;
}