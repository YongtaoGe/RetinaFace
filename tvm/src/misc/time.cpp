//
// Created by geyongtao on 2019-07-09.
//
#include <sys/time.h>
double get_current_time()
{
struct timeval tv;
gettimeofday(&tv, NULL);

return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}