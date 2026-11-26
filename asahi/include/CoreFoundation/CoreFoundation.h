// Minimal CoreFoundation surface for metal-cpp on Linux (types only).
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h> // Apple's CoreFoundation brings in math.h; metal-cpp relies on INFINITY
#ifdef __cplusplus
extern "C" {
#endif
typedef double CFTimeInterval;
typedef const void* CFTypeRef;
typedef const struct __CFString* CFStringRef;
typedef long CFIndex;
typedef struct { CFIndex location, length; } CFRange;
CFTimeInterval CACurrentMediaTime(void);
// Clang lowers __builtin___CFStringMakeConstantString to a constant object
// whose isa is this symbol; libmetal_asahi defines it as the string class.
extern int __CFConstantStringClassReference[];
CFStringRef __CFStringMakeConstantString(const char* cStr);
#ifdef __cplusplus
}
#endif
