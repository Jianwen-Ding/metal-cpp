#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void* _Block_copy(const void* block);
void _Block_release(const void* block);
#ifdef __cplusplus
}
#endif
#define Block_copy(b) ((__typeof(b))_Block_copy((const void*)(b)))
#define Block_release(b) _Block_release((const void*)(b))
