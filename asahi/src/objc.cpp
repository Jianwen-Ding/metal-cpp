// libmetal_asahi: Objective-C runtime subset (selectors, classes, message
// lookup, retain/release, autorelease pools) and the Clang Blocks runtime.
// The block layout and flags follow Clang's published Block ABI
// (clang/docs/Block-ABI-Apple.rst).
#include "rt.h"

#include <cstring>
#include <mutex>
#include <shared_mutex>

namespace masahi {
void ensureRuntime();

static std::mutex& selMutex()
{
    static std::mutex m;
    return m;
}

void fatal(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "metal-asahi: fatal: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    abort();
}

void warn(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "metal-asahi: ");
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

int debugLevel()
{
    static int level = [] {
        const char* v = getenv("METAL_ASAHI_DEBUG");
        return v ? atoi(v) : 0;
    }();
    return level;
}

// ---------------------------------------------------------------- classes

static std::unordered_map<std::string, Class>& classTable()
{
    static auto* t = new std::unordered_map<std::string, Class>();
    return *t;
}
static std::recursive_mutex& classMutex()
{
    static std::recursive_mutex m;
    return m;
}
static Class gRootMeta = nullptr;

Class defineClass(const char* name, Class super)
{
    std::lock_guard<std::recursive_mutex> l(classMutex());
    auto it = classTable().find(name);
    if (it != classTable().end()) return it->second;
    Class meta = new objc_class{nullptr, super ? super->isa : nullptr, strdup((std::string(name) + "(meta)").c_str()),
                                new std::unordered_map<SEL, IMP>(), true};
    Class cls = new objc_class{meta, super, strdup(name), new std::unordered_map<SEL, IMP>(), false};
    if (!gRootMeta) gRootMeta = meta; // the first class defined is NSObject
    meta->isa = gRootMeta;
    if (!super) meta->superclass = cls; // root metaclass inherits the root class's instance methods
    classTable()[name] = cls;
    return cls;
}

static std::atomic<bool> gRegistered{false};
static thread_local bool tRegistering = false;
static void ensureRegistered()
{
    if (gRegistered.load(std::memory_order_acquire) || tRegistering) return;
    static std::once_flag once;
    std::call_once(once, [] {
        tRegistering = true; // registration itself looks classes up
        registerFoundation();
        registerMetal();
        tRegistering = false;
        gRegistered.store(true, std::memory_order_release);
    });
}
void ensureRuntime() { ensureRegistered(); }

Class lookupClass(const char* name)
{
    ensureRegistered();
    std::lock_guard<std::recursive_mutex> l(classMutex());
    auto it = classTable().find(name);
    return it == classTable().end() ? nullptr : it->second;
}

bool isKindOf(const void* obj, Class cls)
{
    if (!obj || !cls) return false;
    for (Class c = static_cast<const objc_object*>(obj)->isa; c; c = c->superclass) {
        if (c == cls) return true;
        if (c->superclass == c) break;
    }
    return false;
}

static IMP findMethod(Class cls, SEL sel)
{
    for (Class c = cls; c; c = c->superclass) {
        auto it = c->methods->find(sel);
        if (it != c->methods->end()) return it->second;
        if (c->superclass == c) break;
    }
    return nullptr;
}

} // namespace masahi
extern "C" objc_class __CFConstantStringClassReference;
namespace masahi {
bool isStatic(const Object* o)
{
    return o->isa == &__CFConstantStringClassReference || (o->isa && o->isa->meta);
}

void release(Object* o)
{
    if (!o || isStatic(o)) return;
    if (o->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (o->destroy) o->destroy(o);
    }
}

// ---------------------------------------------------------------- autorelease pools

struct PoolStack {
    std::vector<std::vector<Object*>> pools;
    ~PoolStack()
    {
        while (!pools.empty()) {
            auto objs = std::move(pools.back());
            pools.pop_back();
            for (Object* o : objs) release(o);
        }
    }
};
static thread_local PoolStack tPools;

Object* autorelease(Object* o)
{
    if (!o || isStatic(o)) return o;
    if (tPools.pools.empty()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true) && debugLevel() > 0)
            warn("object autoreleased with no pool in place (leaked, as on Apple platforms)");
        return o;
    }
    tPools.pools.back().push_back(o);
    return o;
}

void poolPush() { tPools.pools.emplace_back(); }
void poolPop()
{
    if (tPools.pools.empty()) return;
    // Releases can autorelease more objects into this same pool; drain until empty.
    while (!tPools.pools.back().empty()) {
        auto objs = std::move(tPools.pools.back());
        tPools.pools.back().clear();
        for (Object* o : objs) release(o);
    }
    tPools.pools.pop_back();
}

} // namespace masahi

using namespace masahi;

// ---------------------------------------------------------------- C entry points

#pragma GCC visibility push(default)
extern "C" {

SEL sel_registerName(const char* name)
{
    static auto* table = new std::unordered_map<std::string, const char*>();
    std::lock_guard<std::mutex> l(selMutex());
    auto it = table->find(name);
    if (it != table->end()) return reinterpret_cast<SEL>(it->second);
    const char* interned = strdup(name);
    (*table)[name] = interned;
    return reinterpret_cast<SEL>(interned);
}

const char* sel_getName(SEL sel) { return reinterpret_cast<const char*>(sel); }

Class objc_lookUpClass(const char* name) { return lookupClass(name); }
Class objc_getClass(const char* name) { return lookupClass(name); }
Protocol* objc_getProtocol(const char* name)
{
    // Protocols are only compared for identity by metal-cpp; give each name a unique token.
    static std::mutex m;
    static auto* table = new std::unordered_map<std::string, Protocol*>();
    std::lock_guard<std::mutex> l(m);
    auto& p = (*table)[name];
    if (!p) p = new Protocol{nullptr};
    return p;
}

Class object_getClass(id obj) { return obj ? obj->isa : nullptr; }
const char* class_getName(Class cls) { return cls ? cls->name : "nil"; }
Class class_getSuperclass(Class cls) { return cls ? cls->superclass : nullptr; }
BOOL class_respondsToSelector(Class cls, SEL sel) { return cls && findMethod(cls, sel) ? YES : NO; }
IMP class_getMethodImplementation(Class cls, SEL sel) { return cls ? findMethod(cls, sel) : nullptr; }
BOOL class_addMethod(Class cls, SEL name, IMP imp, const char*)
{
    if (!cls || cls->methods->count(name)) return NO;
    (*cls->methods)[name] = imp;
    return YES;
}

static void unrecognized(id self, SEL sel)
{
    const char* cname = self->isa ? self->isa->name : "?";
    fatal("%c[%s %s]: unrecognized selector sent to %s %p (not implemented by metal-asahi)",
          self->isa && self->isa->meta ? '+' : '-', cname, sel_getName(sel),
          self->isa && self->isa->meta ? "class" : "instance", (void*)self);
}

// Called by objc_msgSend with the receiver (non-nil) and the selector.
IMP metal_asahi_objc_lookup(id self, SEL sel)
{
    ensureRuntime();
    IMP imp = findMethod(self->isa, sel);
    if (!imp) unrecognized(self, sel);
    if (debugLevel() >= 3) fprintf(stderr, "metal-asahi: msg [%s %s]\n", self->isa->name, sel_getName(sel));
    return imp;
}

// ---------------------------------------------------------------- Blocks runtime

enum {
    BLOCK_DEALLOCATING = 0x0001,
    BLOCK_REFCOUNT_MASK = 0xfffe,
    BLOCK_NEEDS_FREE = (1 << 24),
    BLOCK_HAS_COPY_DISPOSE = (1 << 25),
    BLOCK_IS_GLOBAL = (1 << 28),
};
enum {
    BLOCK_FIELD_IS_OBJECT = 3,
    BLOCK_FIELD_IS_BLOCK = 7,
    BLOCK_FIELD_IS_BYREF = 8,
    BLOCK_FIELD_IS_WEAK = 16,
    BLOCK_BYREF_CALLER = 128,
};

void* _NSConcreteStackBlock[32];
void* _NSConcreteGlobalBlock[32];
void* _NSConcreteMallocBlock[32];
void* _NSConcreteAutoBlock[32];
void* _NSConcreteFinalizingBlock[32];
void* _NSConcreteWeakBlockVariable[32];

struct BlockDescriptor {
    unsigned long reserved;
    unsigned long size;
    void (*copy)(void* dst, const void* src);
    void (*dispose)(const void*);
};
struct Block {
    void* isa;
    std::atomic<int> flags;
    int reserved;
    void (*invoke)(void*, ...);
    BlockDescriptor* descriptor;
};
struct Byref {
    void* isa;
    Byref* forwarding;
    std::atomic<int> flags;
    int size;
    void (*keep)(Byref* dst, Byref* src);
    void (*destroy)(Byref*);
};

void* _Block_copy(const void* arg)
{
    if (!arg) return nullptr;
    Block* b = (Block*)arg;
    int f = b->flags.load();
    if (f & BLOCK_IS_GLOBAL) return b;
    if (f & BLOCK_NEEDS_FREE) {
        b->flags.fetch_add(2);
        return b;
    }
    Block* n = (Block*)malloc(b->descriptor->size);
    memcpy((void*)n, (const void*)b, b->descriptor->size);
    n->isa = _NSConcreteMallocBlock;
    n->flags.store((f & ~(BLOCK_REFCOUNT_MASK | BLOCK_DEALLOCATING)) | BLOCK_NEEDS_FREE | 2);
    if (f & BLOCK_HAS_COPY_DISPOSE) b->descriptor->copy(n, b);
    return n;
}

void _Block_release(const void* arg)
{
    if (!arg) return;
    Block* b = (Block*)arg;
    int f = b->flags.load();
    if ((f & BLOCK_IS_GLOBAL) || !(f & BLOCK_NEEDS_FREE)) return;
    if ((b->flags.fetch_sub(2) & BLOCK_REFCOUNT_MASK) == 2) {
        if (f & BLOCK_HAS_COPY_DISPOSE) b->descriptor->dispose(b);
        free(b);
    }
}

static Byref* byrefCopy(Byref* src)
{
    if (src->forwarding->flags.load() & BLOCK_NEEDS_FREE) {
        src->forwarding->flags.fetch_add(2);
        return src->forwarding;
    }
    Byref* copy = (Byref*)malloc(src->size);
    copy->isa = nullptr;
    copy->flags.store(src->flags.load() | BLOCK_NEEDS_FREE | 4); // one for the caller, one for the stack
    copy->forwarding = copy;
    src->forwarding = copy;
    copy->size = src->size;
    if (src->flags.load() & BLOCK_HAS_COPY_DISPOSE) {
        copy->keep = src->keep;
        copy->destroy = src->destroy;
        src->keep(copy, src);
    } else {
        memcpy((char*)copy + sizeof(Byref) - 2 * sizeof(void*), (char*)src + sizeof(Byref) - 2 * sizeof(void*),
               src->size - (sizeof(Byref) - 2 * sizeof(void*)));
    }
    return copy;
}

static void byrefRelease(Byref* b)
{
    b = b->forwarding;
    if (!(b->flags.load() & BLOCK_NEEDS_FREE)) return;
    if ((b->flags.fetch_sub(2) & BLOCK_REFCOUNT_MASK) == 2) {
        if (b->flags.load() & BLOCK_HAS_COPY_DISPOSE) b->destroy(b);
        free(b);
    }
}

void _Block_object_assign(void* dst, const void* object, int flags)
{
    void** d = (void**)dst;
    switch (flags & (BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_BLOCK | BLOCK_FIELD_IS_OBJECT)) {
    case BLOCK_FIELD_IS_BYREF: *d = byrefCopy((Byref*)object); break;
    case BLOCK_FIELD_IS_BLOCK: *d = _Block_copy(object); break;
    case BLOCK_FIELD_IS_OBJECT:
        if (!(flags & BLOCK_BYREF_CALLER)) retain((Object*)object);
        *d = (void*)object;
        break;
    default: *d = (void*)object;
    }
}

void _Block_object_dispose(const void* object, int flags)
{
    switch (flags & (BLOCK_FIELD_IS_BYREF | BLOCK_FIELD_IS_BLOCK | BLOCK_FIELD_IS_OBJECT)) {
    case BLOCK_FIELD_IS_BYREF: byrefRelease((Byref*)object); break;
    case BLOCK_FIELD_IS_BLOCK: _Block_release(object); break;
    case BLOCK_FIELD_IS_OBJECT:
        if (!(flags & BLOCK_BYREF_CALLER)) release((Object*)object);
        break;
    default: break;
    }
}

} // extern "C"
#pragma GCC visibility pop
