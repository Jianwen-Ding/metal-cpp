// metal-cpp on Asahi Linux: the Objective-C runtime interface metal-cpp uses.
// Implemented by libmetal_asahi (asahi/src/objc.cpp). Written from the public
// Objective-C ABI shape (isa-first objects, interned selectors); no Apple code.
#pragma once
#if defined(__aarch64__) && !defined(__arm64__)
#define __arm64__ 1 // metal-cpp keys its AAPCS64 objc_msgSend path on __arm64__
#endif
#include <TargetConditionals.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct objc_class* Class;
struct objc_object { Class isa; };
typedef struct objc_object* id;
typedef const struct objc_selector* SEL;
typedef void (*IMP)(void);
typedef struct objc_object Protocol;
typedef signed char BOOL;
#ifndef YES
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#endif
#ifndef Nil
#define Nil ((Class)0)
#endif
#ifndef nil
#define nil ((id)0)
#endif
SEL sel_registerName(const char* name);
const char* sel_getName(SEL sel);
Class objc_lookUpClass(const char* name);
Class objc_getClass(const char* name);
Protocol* objc_getProtocol(const char* name);
Class object_getClass(id obj);
const char* class_getName(Class cls);
Class class_getSuperclass(Class cls);
BOOL class_respondsToSelector(Class cls, SEL sel);
IMP class_getMethodImplementation(Class cls, SEL sel);
BOOL class_addMethod(Class cls, SEL name, IMP imp, const char* types);
#ifdef __cplusplus
}
#endif
