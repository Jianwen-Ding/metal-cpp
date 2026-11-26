#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct dispatch_queue_s* dispatch_queue_t;
typedef struct dispatch_data_s* dispatch_data_t;
typedef uint64_t dispatch_time_t;
#define DISPATCH_TIME_NOW (0ull)
#define DISPATCH_TIME_FOREVER (~0ull)
#define DISPATCH_DATA_DESTRUCTOR_DEFAULT NULL
// dispatch_data_t is backed by libmetal_asahi: bytes are copied.
dispatch_data_t dispatch_data_create(const void* buffer, size_t size, dispatch_queue_t queue, void* destructor);
size_t dispatch_data_get_size(dispatch_data_t data);
const void* metal_asahi_dispatch_data_bytes(dispatch_data_t data);
void dispatch_release(void* object);
#ifdef __cplusplus
}
#endif
