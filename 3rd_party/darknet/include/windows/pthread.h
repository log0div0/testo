
#pragma once

#include <threads.h>

#define pthread_t thrd_t

#define pthread_join thrd_join
#define pthread_create(thr, null, func, arg) thrd_create(thr, func, arg)
