#pragma once
#include <objc/runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// Trampolines (asahi/src/objc_msgsend.S): look up the IMP and tail-call it with
// every argument register intact, so any AAPCS64 signature works.
void objc_msgSend(void);
void objc_msgSend_stret(void);
void objc_msgSend_fpret(void);
#ifdef __cplusplus
}
#endif
