// libmetal_asahi: the Foundation subset metal-cpp programs use.
#include "rt.h"

#include <cmath>
#include <cstring>
#include <ctime>
#include <limits.h>
#include <mutex>
#include <unistd.h>

using namespace masahi;

namespace masahi {
void poolPush();
void poolPop();
}

// Clang's constant NSString/CFString layout (what __builtin___CFStringMakeConstantString emits).
struct ConstString {
    Class isa;
    int flags;
    const char* str;
    long length;
};

extern "C" {
// The isa of every clang constant string. Filled in when Foundation registers.
__attribute__((visibility("default"))) objc_class __CFConstantStringClassReference;
}

namespace masahi {

struct String : Object {
    std::string s;
};

struct Error : Object {
    Ref<Object> domain; // NSString
    intptr_t code = 0;
    Ref<Object> userInfo;
    std::string description;
};

struct URL : Object {
    std::string path;
};

struct Array : Object {
    std::vector<Object*> items;
    ~Array()
    {
        for (Object* o : items) release(o);
    }
};

struct Dictionary : Object {
    std::vector<std::pair<Object*, Object*>> items;
    ~Dictionary()
    {
        for (auto& kv : items) {
            release(kv.first);
            release(kv.second);
        }
    }
};

struct Number : Object {
    enum Kind { I, U, F } kind = I;
    int64_t i = 0;
    uint64_t u = 0;
    double f = 0;
    char type = 'q';
    void* ptr = nullptr;
};

struct Pool : Object {};
struct Bundle : Object {
    std::string path;
};
struct Date : Object {
    double t = 0;
};
struct Lock : Object {
    std::recursive_mutex m;
};

static Class gStringClass, gConstStringClass;

static std::string_view view(const void* o)
{
    if (!o) return {};
    auto* obj = static_cast<const objc_object*>(o);
    if (obj->isa == &__CFConstantStringClassReference) {
        auto* c = static_cast<const ConstString*>(o);
        return std::string_view(c->str, (size_t)c->length);
    }
    if (isKindOf(o, gStringClass)) return static_cast<const String*>(o)->s;
    return {};
}

const char* utf8(const void* o)
{
    if (!o) return nullptr;
    auto* obj = static_cast<const objc_object*>(o);
    if (obj->isa == &__CFConstantStringClassReference) return static_cast<const ConstString*>(o)->str;
    if (isKindOf(o, gStringClass)) return static_cast<const String*>(o)->s.c_str();
    return "";
}
std::string str(const void* o) { return std::string(view(o)); }

Object* newString(const std::string& s)
{
    String* r = make<String>(gStringClass);
    r->s = s;
    return r;
}
Object* tempString(const std::string& s) { return autorelease(newString(s)); }

Object* makeError(const char* domain, intptr_t code, const std::string& description)
{
    Error* e = make<Error>("NSError");
    e->domain = Ref<Object>::adopt(newString(domain));
    e->code = code;
    e->description = description;
    return autorelease(e);
}

void setError(void** out, const char* domain, intptr_t code, const char* fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (debugLevel() > 0) warn("error (%s %ld): %s", domain, (long)code, buf);
    if (out) *out = makeError(domain, code, buf);
}

Object* newArray(const std::vector<Object*>& items)
{
    Array* a = make<Array>("NSArray");
    for (Object* o : items) a->items.push_back(retain(o));
    return a;
}

} // namespace masahi

// NSString constants that metal-cpp looks up with dlsym or links weakly.
#define NS_CONST_STRING(name, text)                                                                        \
    static ConstString name##_storage = {&__CFConstantStringClassReference, 0x7c8, text, sizeof(text) - 1}; \
    extern "C" __attribute__((visibility("default"))) const void* const name = &name##_storage;
NS_CONST_STRING(NSCocoaErrorDomain, "NSCocoaErrorDomain")
NS_CONST_STRING(NSPOSIXErrorDomain, "NSPOSIXErrorDomain")
NS_CONST_STRING(NSOSStatusErrorDomain, "NSOSStatusErrorDomain")
NS_CONST_STRING(NSMachErrorDomain, "NSMachErrorDomain")
NS_CONST_STRING(NSUnderlyingErrorKey, "NSUnderlyingError")
NS_CONST_STRING(NSLocalizedDescriptionKey, "NSLocalizedDescription")
NS_CONST_STRING(NSLocalizedFailureReasonErrorKey, "NSLocalizedFailureReason")
NS_CONST_STRING(NSLocalizedRecoverySuggestionErrorKey, "NSLocalizedRecoverySuggestion")
NS_CONST_STRING(NSLocalizedRecoveryOptionsErrorKey, "NSLocalizedRecoveryOptions")
NS_CONST_STRING(NSRecoveryAttempterErrorKey, "NSRecoveryAttempter")
NS_CONST_STRING(NSHelpAnchorErrorKey, "NSHelpAnchor")
NS_CONST_STRING(NSDebugDescriptionErrorKey, "NSDebugDescription")
NS_CONST_STRING(NSLocalizedFailureErrorKey, "NSLocalizedFailure")
NS_CONST_STRING(NSStringEncodingErrorKey, "NSStringEncodingErrorKey")
NS_CONST_STRING(NSURLErrorKey, "NSURL")
NS_CONST_STRING(NSFilePathErrorKey, "NSFilePath")
NS_CONST_STRING(NSBundleDidLoadNotification, "NSBundleDidLoadNotification")
NS_CONST_STRING(NSBundleResourceRequestLowDiskSpaceNotification, "NSBundleResourceRequestLowDiskSpaceNotification")
NS_CONST_STRING(NSProcessInfoThermalStateDidChangeNotification, "NSProcessInfoThermalStateDidChangeNotification")
NS_CONST_STRING(NSProcessInfoPowerStateDidChangeNotification, "NSProcessInfoPowerStateDidChangeNotification")
NS_CONST_STRING(NSProcessInfoPerformanceProfileDidChangeNotification, "NSProcessInfoPerformanceProfileDidChangeNotification")

extern "C" __attribute__((visibility("default"))) const void* __CFStringMakeConstantString(const char* s)
{
    static std::mutex m;
    static auto* table = new std::unordered_map<std::string, ConstString*>();
    std::lock_guard<std::mutex> l(m);
    auto& c = (*table)[s];
    if (!c) c = new ConstString{&__CFConstantStringClassReference, 0x7c8, strdup(s), (long)strlen(s)};
    return c;
}

extern "C" __attribute__((visibility("default"))) double CACurrentMediaTime(void)
{
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- NSObject

template <typename T> static void defAlloc(Class cls)
{
    defClass(cls, "alloc", +[](Class c, SEL) -> Object* { return make<T>(c); });
    defClass(cls, "allocWithZone:", +[](Class c, SEL, void*) -> Object* { return make<T>(c); });
    defClass(cls, "new", +[](Class c, SEL) -> Object* { return make<T>(c); });
}

static void registerObject(Class o)
{
    defAlloc<Object>(o);
    def(o, "init", +[](Object* self, SEL) { return self; });
    def(o, "retain", +[](Object* self, SEL) { return retain(self); });
    def(o, "release", +[](Object* self, SEL) { release(self); });
    def(o, "autorelease", +[](Object* self, SEL) { return autorelease(self); });
    def(o, "retainCount", +[](Object* self, SEL) -> uintptr_t { return (uintptr_t)self->refcount.load(); });
    def(o, "dealloc", +[](Object* self, SEL) {
        if (self->destroy) self->destroy(self);
    });
    def(o, "hash", +[](Object* self, SEL) -> uintptr_t { return (uintptr_t)self; });
    def(o, "isEqual:", +[](Object* self, SEL, Object* other) -> bool { return self == other; });
    def(o, "copy", +[](Object* self, SEL) { return retain(self); });
    def(o, "class", +[](Object* self, SEL) -> Class { return self->isa; });
    def(o, "respondsToSelector:", +[](Object* self, SEL, SEL s) -> bool { return class_respondsToSelector(self->isa, s); });
    def(o, "methodSignatureForSelector:", +[](Object*, SEL, SEL) -> void* { return nullptr; });
    def(o, "isKindOfClass:", +[](Object* self, SEL, Class c) -> bool { return isKindOf(self, c); });
    def(o, "description", +[](Object* self, SEL) -> Object* {
        char buf[128];
        snprintf(buf, sizeof buf, "<%s: %p>", self->isa->name, (void*)self);
        return tempString(buf);
    });
    def(o, "debugDescription", +[](Object* self, SEL) -> Object* {
        return reinterpret_cast<Object* (*)(Object*, SEL)>(class_getMethodImplementation(self->isa, sel_registerName("description")))(
            self, sel_registerName("description"));
    });
    // Class objects answer the same root messages (they are never freed).
    Class meta = o->isa;
    defClass(o, "retain", +[](Object* self, SEL) { return self; });
    defClass(o, "release", +[](Object*, SEL) {});
    defClass(o, "autorelease", +[](Object* self, SEL) { return self; });
    defClass(o, "class", +[](Object* self, SEL) -> Object* { return self; });
    (void)meta;
}

// ---------------------------------------------------------------- NSString

static Object* stringInit(String* self, const char* s, size_t n)
{
    self->s.assign(s ? s : "", s ? n : 0);
    return self;
}

static void registerString(Class s)
{
    defAlloc<String>(s);
    defClass(s, "string", +[](Class, SEL) -> Object* { return tempString(""); });
    defClass(s, "stringWithString:", +[](Class, SEL, Object* o) -> Object* { return tempString(str(o)); });
    defClass(s, "stringWithCString:encoding:", +[](Class, SEL, const char* c, uintptr_t) -> Object* {
        return c ? tempString(c) : nullptr;
    });
    defClass(s, "stringWithUTF8String:", +[](Class, SEL, const char* c) -> Object* { return c ? tempString(c) : nullptr; });
    def(s, "init", +[](String* self, SEL) { return stringInit(self, "", 0); });
    def(s, "initWithString:", +[](String* self, SEL, Object* o) {
        auto v = view(o);
        return stringInit(self, v.data(), v.size());
    });
    def(s, "initWithCString:encoding:", +[](String* self, SEL, const char* c, uintptr_t) {
        return stringInit(self, c, c ? strlen(c) : 0);
    });
    def(s, "initWithUTF8String:", +[](String* self, SEL, const char* c) { return stringInit(self, c, c ? strlen(c) : 0); });
    def(s, "initWithBytesNoCopy:length:encoding:freeWhenDone:", +[](String* self, SEL, void* b, uintptr_t n, uintptr_t, bool f) {
        stringInit(self, (const char*)b, n);
        if (f) free(b);
        return (Object*)self;
    });
    def(s, "UTF8String", +[](Object* self, SEL) { return utf8(self); });
    def(s, "cStringUsingEncoding:", +[](Object* self, SEL, uintptr_t) { return utf8(self); });
    def(s, "fileSystemRepresentation", +[](Object* self, SEL) { return utf8(self); });
    def(s, "length", +[](Object* self, SEL) -> uintptr_t { return view(self).size(); });
    def(s, "lengthOfBytesUsingEncoding:", +[](Object* self, SEL, uintptr_t) -> uintptr_t { return view(self).size(); });
    def(s, "maximumLengthOfBytesUsingEncoding:", +[](Object* self, SEL, uintptr_t) -> uintptr_t { return view(self).size() * 3 + 1; });
    def(s, "characterAtIndex:", +[](Object* self, SEL, uintptr_t i) -> unsigned short {
        auto v = view(self);
        return i < v.size() ? (unsigned char)v[i] : 0;
    });
    def(s, "isEqualToString:", +[](Object* self, SEL, Object* o) -> bool { return o && view(self) == view(o); });
    def(s, "isEqual:", +[](Object* self, SEL, Object* o) -> bool {
        return o && (isKindOf(o, gStringClass)) && view(self) == view(o);
    });
    def(s, "hash", +[](Object* self, SEL) -> uintptr_t { return std::hash<std::string_view>()(view(self)); });
    def(s, "stringByAppendingString:", +[](Object* self, SEL, Object* o) -> Object* { return tempString(str(self) + str(o)); });
    def(s, "caseInsensitiveCompare:", +[](Object* self, SEL, Object* o) -> intptr_t {
        int r = strcasecmp(utf8(self), o ? utf8(o) : "");
        return r < 0 ? -1 : r > 0 ? 1 : 0;
    });
    def(s, "compare:", +[](Object* self, SEL, Object* o) -> intptr_t {
        int r = view(self).compare(view(o));
        return r < 0 ? -1 : r > 0 ? 1 : 0;
    });
    struct NSRange { uintptr_t location, length; };
    def(s, "rangeOfString:options:", +[](Object* self, SEL, Object* o, uintptr_t) -> NSRange {
        auto pos = view(self).find(view(o));
        if (pos == std::string_view::npos) return {(uintptr_t)LONG_MAX, 0};
        return {pos, view(o).size()};
    });
    def(s, "description", +[](Object* self, SEL) { return self; });
    def(s, "copy", +[](Object* self, SEL) -> Object* {
        if (self->isa == &__CFConstantStringClassReference) return self;
        return newString(str(self));
    });
}

// ---------------------------------------------------------------- the rest

static void registerOthers()
{
    Class err = defineClass("NSError", lookupClass("NSObject"));
    defAlloc<Error>(err);
    auto initErr = +[](Error* self, SEL, Object* domain, intptr_t code, Object* info) -> Object* {
        self->domain = Ref<Object>(domain);
        self->code = code;
        self->userInfo = Ref<Object>(info);
        return self;
    };
    def(err, "initWithDomain:code:userInfo:", initErr);
    defClass(err, "errorWithDomain:code:userInfo:", +[](Class c, SEL, Object* d, intptr_t code, Object* info) -> Object* {
        Error* e = make<Error>(c);
        e->domain = Ref<Object>(d);
        e->code = code;
        e->userInfo = Ref<Object>(info);
        return autorelease(e);
    });
    def(err, "domain", +[](Error* self, SEL) { return self->domain.get(); });
    def(err, "code", +[](Error* self, SEL) { return self->code; });
    def(err, "userInfo", +[](Error* self, SEL) { return self->userInfo.get(); });
    def(err, "localizedDescription", +[](Error* self, SEL) -> Object* {
        if (!self->description.empty()) return tempString(self->description);
        char buf[256];
        snprintf(buf, sizeof buf, "The operation couldn't be completed. (%s error %ld.)", utf8(self->domain.get()), (long)self->code);
        return tempString(buf);
    });
    def(err, "localizedFailureReason", +[](Error* self, SEL) -> Object* { return tempString(self->description); });
    def(err, "localizedRecoverySuggestion", +[](Error*, SEL) -> Object* { return nullptr; });
    def(err, "localizedRecoveryOptions", +[](Error*, SEL) -> Object* { return nullptr; });
    def(err, "description", +[](Error* self, SEL) -> Object* {
        return tempString("Error Domain=" + str(self->domain.get()) + " Code=" + std::to_string(self->code) + " \"" +
                          self->description + "\"");
    });

    Class url = defineClass("NSURL", lookupClass("NSObject"));
    defAlloc<URL>(url);
    defClass(url, "fileURLWithPath:", +[](Class c, SEL, Object* p) -> Object* {
        URL* u = make<URL>(c);
        u->path = str(p);
        return autorelease(u);
    });
    def(url, "initFileURLWithPath:", +[](URL* self, SEL, Object* p) -> Object* {
        self->path = str(p);
        return self;
    });
    def(url, "fileSystemRepresentation", +[](URL* self, SEL) { return self->path.c_str(); });
    def(url, "path", +[](URL* self, SEL) -> Object* { return tempString(self->path); });
    def(url, "description", +[](URL* self, SEL) -> Object* { return tempString("file://" + self->path); });

    Class arr = defineClass("NSArray", lookupClass("NSObject"));
    defAlloc<Array>(arr);
    defClass(arr, "array", +[](Class c, SEL) -> Object* { return autorelease(make<Array>(c)); });
    defClass(arr, "arrayWithObject:", +[](Class, SEL, Object* o) -> Object* { return autorelease(newArray({o})); });
    defClass(arr, "arrayWithObjects:count:", +[](Class, SEL, Object** o, uintptr_t n) -> Object* {
        return autorelease(newArray(std::vector<Object*>(o, o + n)));
    });
    def(arr, "initWithObjects:count:", +[](Array* self, SEL, Object** o, uintptr_t n) -> Object* {
        for (uintptr_t i = 0; i < n; i++) self->items.push_back(retain(o[i]));
        return self;
    });
    def(arr, "count", +[](Array* self, SEL) -> uintptr_t { return self->items.size(); });
    def(arr, "objectAtIndex:", +[](Array* self, SEL, uintptr_t i) -> Object* {
        if (i >= self->items.size()) fatal("-[NSArray objectAtIndex:]: index %zu beyond bounds [0 .. %zu]", (size_t)i, self->items.size());
        return self->items[i];
    });
    def(arr, "object:", +[](Array* self, SEL, uintptr_t i) -> Object* { return i < self->items.size() ? self->items[i] : nullptr; });
    struct FastEnum {
        unsigned long state;
        Object** itemsPtr;
        unsigned long* mutationsPtr;
        unsigned long extra[5];
    };
    def(arr, "countByEnumeratingWithState:objects:count:", +[](Array* self, SEL, FastEnum* st, Object**, uintptr_t) -> uintptr_t {
        static unsigned long mutations = 0;
        if (st->state) return 0;
        st->state = 1;
        st->itemsPtr = self->items.data();
        st->mutationsPtr = &mutations;
        return self->items.size();
    });

    Class dict = defineClass("NSDictionary", lookupClass("NSObject"));
    defAlloc<Dictionary>(dict);
    defClass(dict, "dictionary", +[](Class c, SEL) -> Object* { return autorelease(make<Dictionary>(c)); });
    defClass(dict, "dictionaryWithObject:forKey:", +[](Class c, SEL, Object* v, Object* k) -> Object* {
        Dictionary* d = make<Dictionary>(c);
        d->items.push_back({retain(k), retain(v)});
        return autorelease(d);
    });
    defClass(dict, "dictionaryWithObjects:forKeys:count:", +[](Class c, SEL, Object** v, Object** k, uintptr_t n) -> Object* {
        Dictionary* d = make<Dictionary>(c);
        for (uintptr_t i = 0; i < n; i++) d->items.push_back({retain(k[i]), retain(v[i])});
        return autorelease(d);
    });
    def(dict, "initWithObjects:forKeys:count:", +[](Dictionary* d, SEL, Object** v, Object** k, uintptr_t n) -> Object* {
        for (uintptr_t i = 0; i < n; i++) d->items.push_back({retain(k[i]), retain(v[i])});
        return d;
    });
    def(dict, "count", +[](Dictionary* d, SEL) -> uintptr_t { return d->items.size(); });
    def(dict, "objectForKey:", +[](Dictionary* d, SEL, Object* k) -> Object* {
        for (auto& kv : d->items) {
            if (kv.first == k) return kv.second;
            if (isKindOf(k, gStringClass) && isKindOf(kv.first, gStringClass) && view(k) == view(kv.first)) return kv.second;
        }
        return nullptr;
    });

    Class val = defineClass("NSValue", lookupClass("NSObject"));
    Class num = defineClass("NSNumber", val);
    defAlloc<Number>(val);
    defAlloc<Number>(num);
#define NUM_CTOR(selName, initName, T, K, code)                                                       \
    defClass(num, selName, +[](Class c, SEL, T v) -> Object* {                                         \
        Number* n = make<Number>(c);                                                                   \
        n->kind = Number::K;                                                                           \
        n->i = (int64_t)v; n->u = (uint64_t)v; n->f = (double)v; n->type = code;                        \
        return autorelease(n);                                                                         \
    });                                                                                                \
    def(num, initName, +[](Number* n, SEL, T v) -> Object* {                                           \
        n->kind = Number::K;                                                                           \
        n->i = (int64_t)v; n->u = (uint64_t)v; n->f = (double)v; n->type = code;                        \
        return n;                                                                                      \
    });
    NUM_CTOR("numberWithChar:", "initWithChar:", char, I, 'c')
    NUM_CTOR("numberWithUnsignedChar:", "initWithUnsignedChar:", unsigned char, U, 'C')
    NUM_CTOR("numberWithShort:", "initWithShort:", short, I, 's')
    NUM_CTOR("numberWithUnsignedShort:", "initWithUnsignedShort:", unsigned short, U, 'S')
    NUM_CTOR("numberWithInt:", "initWithInt:", int, I, 'i')
    NUM_CTOR("numberWithUnsignedInt:", "initWithUnsignedInt:", unsigned int, U, 'I')
    NUM_CTOR("numberWithLong:", "initWithLong:", long, I, 'q')
    NUM_CTOR("numberWithUnsignedLong:", "initWithUnsignedLong:", unsigned long, U, 'Q')
    NUM_CTOR("numberWithLongLong:", "initWithLongLong:", long long, I, 'q')
    NUM_CTOR("numberWithUnsignedLongLong:", "initWithUnsignedLongLong:", unsigned long long, U, 'Q')
    NUM_CTOR("numberWithFloat:", "initWithFloat:", float, F, 'f')
    NUM_CTOR("numberWithDouble:", "initWithDouble:", double, F, 'd')
    NUM_CTOR("numberWithBool:", "initWithBool:", bool, U, 'B')
#undef NUM_CTOR
#define NUM_GET(selName, T)                                                                            \
    def(num, selName, +[](Number* n, SEL) -> T {                                                       \
        return n->kind == Number::F ? (T)n->f : n->kind == Number::U ? (T)n->u : (T)n->i;              \
    });
    NUM_GET("charValue", char)
    NUM_GET("unsignedCharValue", unsigned char)
    NUM_GET("shortValue", short)
    NUM_GET("unsignedShortValue", unsigned short)
    NUM_GET("intValue", int)
    NUM_GET("unsignedIntValue", unsigned int)
    NUM_GET("longValue", long)
    NUM_GET("unsignedLongValue", unsigned long)
    NUM_GET("longLongValue", long long)
    NUM_GET("unsignedLongLongValue", unsigned long long)
    NUM_GET("integerValue", intptr_t)
    NUM_GET("unsignedIntegerValue", uintptr_t)
    NUM_GET("floatValue", float)
    NUM_GET("doubleValue", double)
    NUM_GET("boolValue", bool)
#undef NUM_GET
    def(num, "isEqualToNumber:", +[](Number* a, SEL, Number* b) -> bool { return b && a->f == b->f && a->i == b->i; });
    def(num, "stringValue", +[](Number* n, SEL) -> Object* {
        return tempString(n->kind == Number::F ? std::to_string(n->f) : n->kind == Number::U ? std::to_string(n->u) : std::to_string(n->i));
    });
    def(num, "description", +[](Number* n, SEL) -> Object* {
        return tempString(n->kind == Number::F ? std::to_string(n->f) : n->kind == Number::U ? std::to_string(n->u) : std::to_string(n->i));
    });
    defClass(val, "valueWithPointer:", +[](Class c, SEL, void* p) -> Object* {
        Number* n = make<Number>(c);
        n->ptr = p;
        n->type = '^';
        return autorelease(n);
    });
    def(val, "pointerValue", +[](Number* n, SEL) { return n->ptr; });

    Class pool = defineClass("NSAutoreleasePool", lookupClass("NSObject"));
    defClass(pool, "alloc", +[](Class c, SEL) -> Object* { return make<Pool>(c); });
    def(pool, "init", +[](Pool* self, SEL) -> Object* {
        poolPush();
        return self;
    });
    auto drain = +[](Pool* self, SEL) {
        poolPop();
        self->destroy(self);
    };
    def(pool, "drain", drain);
    def(pool, "release", drain);
    def(pool, "retain", +[](Pool* self, SEL) -> Object* { return self; });
    def(pool, "autorelease", +[](Pool* self, SEL) -> Object* { return self; });
    defClass(pool, "showPools", +[](Class, SEL) {});

    Class bundle = defineClass("NSBundle", lookupClass("NSObject"));
    defAlloc<Bundle>(bundle);
    defClass(bundle, "mainBundle", +[](Class c, SEL) -> Object* {
        static Bundle* main = [&] {
            Bundle* b = make<Bundle>(c);
            char exe[PATH_MAX];
            ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
            std::string p = n > 0 ? std::string(exe, (size_t)n) : ".";
            b->path = p.substr(0, p.find_last_of('/'));
            return b;
        }();
        return main;
    });
    def(bundle, "initWithPath:", +[](Bundle* self, SEL, Object* p) -> Object* {
        self->path = str(p);
        return self;
    });
    defClass(bundle, "bundleWithPath:", +[](Class c, SEL, Object* p) -> Object* {
        Bundle* b = make<Bundle>(c);
        b->path = str(p);
        return autorelease(b);
    });
    def(bundle, "bundlePath", +[](Bundle* self, SEL) -> Object* { return tempString(self->path); });
    def(bundle, "resourcePath", +[](Bundle* self, SEL) -> Object* { return tempString(self->path); });
    def(bundle, "executablePath", +[](Bundle*, SEL) -> Object* {
        char exe[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        return tempString(n > 0 ? std::string(exe, (size_t)n) : "");
    });
    def(bundle, "bundleURL", +[](Bundle* self, SEL) -> Object* {
        URL* u = make<URL>("NSURL");
        u->path = self->path;
        return autorelease(u);
    });
    def(bundle, "resourceURL", +[](Bundle* self, SEL) -> Object* {
        URL* u = make<URL>("NSURL");
        u->path = self->path;
        return autorelease(u);
    });
    def(bundle, "URLForResource:withExtension:", +[](Bundle* self, SEL, Object* name, Object* ext) -> Object* {
        std::string p = self->path + "/" + str(name) + (ext ? "." + str(ext) : "");
        if (access(p.c_str(), R_OK) != 0) return nullptr;
        URL* u = make<URL>("NSURL");
        u->path = p;
        return autorelease(u);
    });
    def(bundle, "bundleIdentifier", +[](Bundle*, SEL) -> Object* { return nullptr; });
    def(bundle, "isLoaded", +[](Bundle*, SEL) -> bool { return true; });

    Class pinfo = defineClass("NSProcessInfo", lookupClass("NSObject"));
    defClass(pinfo, "processInfo", +[](Class c, SEL) -> Object* {
        static Object* p = make<Object>(c);
        return p;
    });
    def(pinfo, "processorCount", +[](Object*, SEL) -> uintptr_t { return (uintptr_t)sysconf(_SC_NPROCESSORS_CONF); });
    def(pinfo, "activeProcessorCount", +[](Object*, SEL) -> uintptr_t { return (uintptr_t)sysconf(_SC_NPROCESSORS_ONLN); });
    def(pinfo, "physicalMemory", +[](Object*, SEL) -> unsigned long long {
        return (unsigned long long)sysconf(_SC_PHYS_PAGES) * (unsigned long long)sysconf(_SC_PAGESIZE);
    });
    def(pinfo, "systemUptime", +[](Object*, SEL) -> double { return CACurrentMediaTime(); });
    def(pinfo, "processIdentifier", +[](Object*, SEL) -> int { return getpid(); });
    def(pinfo, "isLowPowerModeEnabled", +[](Object*, SEL) -> bool { return false; });
    def(pinfo, "thermalState", +[](Object*, SEL) -> intptr_t { return 0; });
    def(pinfo, "processName", +[](Object*, SEL) -> Object* {
        char exe[PATH_MAX];
        ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        std::string p = n > 0 ? std::string(exe, (size_t)n) : "";
        return tempString(p.substr(p.find_last_of('/') + 1));
    });

    Class date = defineClass("NSDate", lookupClass("NSObject"));
    defAlloc<Date>(date);
    defClass(date, "dateWithTimeIntervalSinceNow:", +[](Class c, SEL, double dt) -> Object* {
        Date* d = make<Date>(c);
        d->t = CACurrentMediaTime() + dt;
        return autorelease(d);
    });

    Class lock = defineClass("NSLock", lookupClass("NSObject"));
    defAlloc<Lock>(lock);
    def(lock, "lock", +[](Lock* l, SEL) { l->m.lock(); });
    def(lock, "unlock", +[](Lock* l, SEL) { l->m.unlock(); });
}

void masahi::registerFoundation()
{
    Class o = defineClass("NSObject", nullptr);
    registerObject(o);
    gStringClass = defineClass("NSString", o);
    registerString(gStringClass);
    // The constant-string class is statically allocated so clang's constant
    // strings can point at it before anything runs.
    Class meta = new objc_class{o->isa, gStringClass->isa, "__NSCFConstantString(meta)", new std::unordered_map<SEL, IMP>(), true};
    __CFConstantStringClassReference = objc_class{meta, gStringClass, "__NSCFConstantString", new std::unordered_map<SEL, IMP>(), false};
    gConstStringClass = &__CFConstantStringClassReference;
    def(gConstStringClass, "retain", +[](Object* self, SEL) { return self; });
    def(gConstStringClass, "release", +[](Object*, SEL) {});
    def(gConstStringClass, "autorelease", +[](Object* self, SEL) { return self; });
    def(gConstStringClass, "retainCount", +[](Object*, SEL) -> uintptr_t { return UINTPTR_MAX; });
    registerOthers();
}
