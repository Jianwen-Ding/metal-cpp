// The Objective-C runtime and Foundation subset under metal-cpp.
#include "harness.h"
#include <Block.h>

TEST(string_literal)
{
    NS::String* s = MTLSTR("hello metal");
    CHECK(s != nullptr);
    CHECK(strcmp(s->utf8String(), "hello metal") == 0);
    CHECK(s->length() == 11);
    CHECK(s->isEqualToString(NS::String::string("hello metal", NS::UTF8StringEncoding)));
}

TEST(string_create_append)
{
    NS::String* a = NS::String::string("abc", NS::UTF8StringEncoding);
    NS::String* b = a->stringByAppendingString(MTLSTR("def"));
    CHECK(strcmp(b->utf8String(), "abcdef") == 0);
    CHECK(b->length() == 6);
    CHECK(b->character(3) == 'd');
    CHECK(a->caseInsensitiveCompare(MTLSTR("ABC")) == NS::OrderedSame);
    NS::Range r = b->rangeOfString(MTLSTR("cd"), NS::CaseInsensitiveSearch);
    CHECK(r.location == 2 && r.length == 2);
    NS::String* c = NS::String::alloc()->init("xyz", NS::UTF8StringEncoding);
    CHECK(strcmp(c->cString(NS::UTF8StringEncoding), "xyz") == 0);
    c->release();
}

TEST(retain_release)
{
    NS::String* s = NS::String::alloc()->init("counted", NS::UTF8StringEncoding);
    CHECK(s->retainCount() == 1);
    s->retain();
    CHECK(s->retainCount() == 2);
    s->release();
    CHECK(s->retainCount() == 1);
    s->release();
}

TEST(autorelease_pool_nesting)
{
    NS::AutoreleasePool* outer = NS::AutoreleasePool::alloc()->init();
    NS::String* s = NS::String::alloc()->init("pooled", NS::UTF8StringEncoding);
    s->retain();
    s->autorelease();
    {
        NS::AutoreleasePool* inner = NS::AutoreleasePool::alloc()->init();
        NS::String::string("temp", NS::UTF8StringEncoding);
        inner->release();
    }
    CHECK(s->retainCount() == 2);
    outer->release();
    CHECK(s->retainCount() == 1);
    s->release();
}

TEST(number)
{
    NS::Number* n = NS::Number::number(42);
    CHECK(n->intValue() == 42);
    CHECK(n->doubleValue() == 42.0);
    NS::Number* f = NS::Number::number(2.5f);
    CHECK(f->floatValue() == 2.5f);
    CHECK(f->intValue() == 2);
    NS::Number* b = NS::Number::number(true);
    CHECK(b->boolValue());
    NS::Number* u = NS::Number::number((unsigned long long)1 << 40);
    CHECK(u->unsignedLongLongValue() == (1ull << 40));
}

TEST(array)
{
    NS::Object* items[3] = {MTLSTR("a"), MTLSTR("b"), MTLSTR("c")};
    NS::Array* arr = NS::Array::array(items, 3);
    CHECK(arr->count() == 3);
    CHECK(strcmp(arr->object<NS::String>(1)->utf8String(), "b") == 0);
    NS::Array* one = NS::Array::array(MTLSTR("solo"));
    CHECK(one->count() == 1);
}

TEST(dictionary)
{
    NS::Object* keys[2] = {MTLSTR("k1"), MTLSTR("k2")};
    NS::Object* vals[2] = {NS::Number::number(1), NS::Number::number(2)};
    NS::Dictionary* d = NS::Dictionary::dictionary(vals, keys, 2);
    CHECK(d->count() == 2);
    NS::Number* v = d->object<NS::Number>(NS::String::string("k2", NS::UTF8StringEncoding));
    CHECK(v && v->intValue() == 2);
}

TEST(error_object)
{
    NS::Error* e = NS::Error::error(MTLSTR("TestDomain"), 17, nullptr);
    CHECK(e->code() == 17);
    CHECK(strcmp(e->domain()->utf8String(), "TestDomain") == 0);
    CHECK(e->localizedDescription() != nullptr);
}

TEST(url_and_bundle)
{
    NS::URL* u = NS::URL::fileURLWithPath(MTLSTR("/tmp/x.air"));
    CHECK(strcmp(u->fileSystemRepresentation(), "/tmp/x.air") == 0);
    NS::Bundle* b = NS::Bundle::mainBundle();
    CHECK(b != nullptr);
    CHECK(b->bundlePath() != nullptr);
}

TEST(process_info)
{
    NS::ProcessInfo* p = NS::ProcessInfo::processInfo();
    CHECK(p->processorCount() > 0);
    CHECK(p->physicalMemory() > (1ull << 30));
}

TEST(blocks_copy_and_call)
{
    __block int counter = 0;
    void (^inc)(int) = ^(int k) { counter += k; };
    void (^heap)(int) = (void (^)(int))_Block_copy(inc);
    heap(5);
    inc(2);
    _Block_release(heap);
    CHECK(counter == 7);
}

TEST(device_name_string)
{
    NS::String* n = t::device()->name();
    CHECK(n && strstr(n->utf8String(), "Apple") != nullptr);
}

TEST_MAIN
