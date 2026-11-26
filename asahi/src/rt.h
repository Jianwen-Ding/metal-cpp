// libmetal_asahi internal runtime: a small Objective-C-compatible object model.
//
// Objects are C++ structs whose first word is the class pointer (isa), as
// objc_msgSend expects. Methods are plain functions `R f(Self*, SEL, Args...)`
// registered per class under an interned selector; objc_msgSend tail-calls
// them with the caller's argument registers, so they have exactly the AAPCS64
// signature metal-cpp casts objc_msgSend to.
#pragma once

#define METAL_ASAHI_RUNTIME_IMPL 1
#include <objc/runtime.h>

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
void* _Block_copy(const void* block);
void _Block_release(const void* block);
void objc_msgSend(void);
double CACurrentMediaTime(void);
}

struct objc_class {
    Class isa;          // metaclass (class methods); a metaclass points at the root metaclass
    Class superclass;
    const char* name;
    std::unordered_map<SEL, IMP>* methods;
    bool meta;
};

namespace masahi {

[[noreturn]] void fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void warn(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
int debugLevel(); // METAL_ASAHI_DEBUG

struct Object {
    Class isa = nullptr;
    std::atomic<intptr_t> refcount{1};
    void (*destroy)(Object*) = nullptr;
};

// Creates (or returns) the class `name` with superclass `super`.
Class defineClass(const char* name, Class super);
Class lookupClass(const char* name);

template <typename F> inline void def(Class cls, const char* sel, F* fn)
{
    (*cls->methods)[sel_registerName(sel)] = reinterpret_cast<IMP>(fn);
}
template <typename F> inline void defClass(Class cls, const char* sel, F* fn)
{
    (*cls->isa->methods)[sel_registerName(sel)] = reinterpret_cast<IMP>(fn);
}

template <typename T> T* make(Class cls)
{
    T* o = new T();
    o->isa = cls;
    o->destroy = [](Object* p) { delete static_cast<T*>(p); };
    return o;
}
template <typename T> T* make(const char* cls) { return make<T>(lookupClass(cls)); }

// Objects that are not refcounted: clang constant strings and class objects.
bool isStatic(const Object* o);
inline Object* retain(Object* o)
{
    if (o && !isStatic(o)) o->refcount.fetch_add(1, std::memory_order_relaxed);
    return o;
}
void release(Object* o);
Object* autorelease(Object* o);

template <typename T> inline T* retainT(T* o) { return static_cast<T*>(retain(o)); }

// Holds a strong reference.
template <typename T> struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(T* q) : p(q) { retain(p); }
    Ref(const Ref& r) : p(r.p) { retain(p); }
    Ref(Ref&& r) noexcept : p(r.p) { r.p = nullptr; }
    Ref& operator=(const Ref& r) { if (r.p != p) { retain(r.p); release(p); p = r.p; } return *this; }
    Ref& operator=(Ref&& r) noexcept { if (this != &r) { release(p); p = r.p; r.p = nullptr; } return *this; }
    ~Ref() { release(p); }
    static Ref adopt(T* q) { Ref r; r.p = q; return r; }
    T* operator->() const { return p; }
    T* get() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

bool isKindOf(const void* obj, Class cls);
template <typename T> T* cast(const void* obj, const char* cls)
{
    return obj && isKindOf(obj, lookupClass(cls)) ? static_cast<T*>(const_cast<void*>(obj)) : nullptr;
}

// Foundation helpers used across the runtime.
Object* newString(const std::string& s);                // +1
Object* tempString(const std::string& s);               // autoreleased
const char* utf8(const void* nsString);                 // nullptr for nil
std::string str(const void* nsString);
Object* makeError(const char* domain, intptr_t code, const std::string& description); // autoreleased
void setError(void** outError, const char* domain, intptr_t code, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));
Object* newArray(const std::vector<Object*>& items);    // +1, retains items

// Blocks (Clang block ABI).
struct BlockLayout {
    void* isa;
    int flags;
    int reserved;
    void (*invoke)(void*, ...);
    void* descriptor;
};
template <typename... A> inline void callBlock(const void* block, A... a)
{
    auto* b = static_cast<const BlockLayout*>(block);
    reinterpret_cast<void (*)(const void*, A...)>(b->invoke)(block, a...);
}

void registerFoundation();
void registerMetal();

} // namespace masahi
