// libmetal_asahi: AIR -> AGX through the in-tree LLVM AGX backend.
#include "compiler.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <chrono>
#include <cstring>
#include <map>
#include <mutex>

using namespace llvm;

namespace masahi {

static void initLLVM()
{
    static std::once_flag once;
    std::call_once(once, [] {
        LLVMInitializeAGXTargetInfo();
        LLVMInitializeAGXTarget();
        LLVMInitializeAGXTargetMC();
        LLVMInitializeAGXAsmPrinter();
    });
}

static uint32_t rd32(const uint8_t* p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}
static uint64_t rd64(const uint8_t* p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

std::vector<std::vector<uint8_t>> splitAIRContainer(const uint8_t* data, size_t size, std::string& error)
{
    std::vector<std::vector<uint8_t>> out;
    auto takeWrapped = [&](size_t off, size_t limit) -> size_t {
        if (off + 20 > limit || rd32(data + off) != 0x0B17C0DE) return 0;
        uint32_t bcOff = rd32(data + off + 8), bcSize = rd32(data + off + 12);
        if (off + (size_t)bcOff + bcSize > limit) return 0;
        out.emplace_back(data + off, data + off + bcOff + bcSize);
        return bcOff + bcSize;
    };
    if (size >= 4 && rd32(data) == 0x0B17C0DE) {
        if (!takeWrapped(0, size)) error = "truncated AIR bitcode wrapper";
        return out;
    }
    if (size >= 4 && data[0] == 'B' && data[1] == 'C' && data[2] == 0xC0 && data[3] == 0xDE) {
        out.emplace_back(data, data + size); // raw bitcode
        return out;
    }
    if (size >= 88 && memcmp(data, "MTLB", 4) == 0) {
        // The container's bitcode section holds one wrapped module per entry.
        // Header layouts differ between container versions, so find the
        // modules by their wrapper magic, each checked against its own size.
        for (size_t off = 88; off + 20 <= size;) {
            size_t n = rd32(data + off) == 0x0B17C0DE ? takeWrapped(off, size) : 0;
            off += n ? n : 1;
        }
        if (out.empty()) error = "metallib: no AIR bitcode found";
        return out;
    }
    error = "not an AIR file (.air bitcode or .metallib)";
    return out;
}

namespace {

struct DiagCollector : DiagnosticHandler {
    std::string* errors;
    explicit DiagCollector(std::string* e) : errors(e) {}
    bool handleDiagnostics(const DiagnosticInfo& DI) override
    {
        if (DI.getSeverity() == DS_Error) {
            raw_string_ostream os(*errors);
            DiagnosticPrinterRawOStream dp(os);
            DI.print(dp);
            os << "\n";
        }
        return true;
    }
};

unsigned mdInt(const MDOperand& op)
{
    if (auto* c = mdconst::dyn_extract_or_null<ConstantInt>(op)) return (unsigned)c->getZExtValue();
    return ~0u;
}

// Metal binding facts from !air.kernel, before agx-lower-air rewrites the module.
struct AIRKernelInfo {
    std::vector<std::pair<uint32_t, uint32_t>> buffers; // (location, address space) in parameter order
    int tgLocation = -1;
};

std::map<std::string, AIRKernelInfo> readAIRKernels(Module& M)
{
    std::map<std::string, AIRKernelInfo> result;
    NamedMDNode* md = M.getNamedMetadata("air.kernel");
    if (!md) return result;
    for (MDNode* n : md->operands()) {
        if (n->getNumOperands() < 3) continue;
        auto* F = mdconst::dyn_extract_or_null<Function>(n->getOperand(0));
        auto* params = dyn_cast<MDNode>(n->getOperand(2));
        if (!F || !params) continue;
        struct P { unsigned argNo, loc, as; bool buffer; };
        std::vector<P> ps;
        for (const MDOperand& po : params->operands()) {
            auto* p = dyn_cast<MDNode>(po);
            if (!p || p->getNumOperands() < 2) continue;
            auto* role = dyn_cast<MDString>(p->getOperand(1));
            P x{mdInt(p->getOperand(0)), 0, 1, role && role->getString() == "air.buffer"};
            for (unsigned i = 2; i + 1 < p->getNumOperands(); ++i) {
                auto* key = dyn_cast<MDString>(p->getOperand(i));
                if (!key) continue;
                if (key->getString() == "air.location_index") x.loc = mdInt(p->getOperand(++i));
                else if (key->getString() == "air.address_space") x.as = mdInt(p->getOperand(++i));
            }
            ps.push_back(x);
        }
        std::sort(ps.begin(), ps.end(), [](const P& a, const P& b) { return a.argNo < b.argNo; });
        AIRKernelInfo info;
        for (auto& p : ps) {
            if (!p.buffer) continue;
            if (p.as == 3) info.tgLocation = (int)p.loc;
            else info.buffers.push_back({p.loc, p.as});
        }
        result[F->getName().str()] = std::move(info);
    }
    return result;
}

bool parseRecord(const uint8_t* p, size_t n, CompiledKernel& k)
{
    size_t words = n / 4, i = 0;
    auto w = [&](size_t j) { return rd32(p + 4 * j); };
    if (words < 3) return false;
    uint32_t size = w(0), version = w(1), nameLen = w(2);
    if (version != 1 || size > n) return false;
    words = size / 4;
    i = 3;
    k.name.assign((const char*)p + 12, nameLen);
    i += (nameLen + 3) / 4;
    if (i + 6 > words) return false;
    k.uniformHalves = w(i++);
    k.tgpgHalf = w(i++);
    k.globalDataHalf = w(i++);
    k.gprs = w(i++);
    k.threadgroupBytes = w(i++);
    k.stackBytes = w(i++);
    if (i >= words) return false;
    uint32_t nargs = w(i++);
    for (uint32_t a = 0; a < nargs && i + 3 <= words; a++) {
        KernelArg arg;
        arg.kind = w(i++);
        arg.size = w(i++);
        arg.firstHalf = w(i++);
        k.args.push_back(arg);
    }
    while (i < words) {
        uint32_t tag = w(i++);
        switch (tag) {
        case 1:
            if (i + 3 > words) return true;
            for (int d = 0; d < 3; d++) k.clusterDims[d] = w(i++);
            break;
        case 2: { // checked: version, N, record half, tg bytes half, N x 4
            if (i + 4 > words) return true;
            k.checked = true;
            uint32_t ns = w(i + 1);
            i += 4 + 4 * ns;
            break;
        }
        case 3: i += 2; break;
        case 4: k.hiddenTgBytesHalf = w(i++); break;
        case 5: k.textureHeapHalf = w(i++); break;
        case 6: {
            uint32_t c = w(i++);
            for (uint32_t j = 0; j < c && i + 2 <= words; j++, i += 2) k.uniformConstants.push_back({w(i), w(i + 1)});
            break;
        }
        case 7: {
            uint32_t c = w(i++);
            for (uint32_t j = 0; j < c && i + 3 <= words; j++, i += 3) k.divisors.push_back({w(i), w(i + 1), w(i + 2)});
            break;
        }
        case 8:
            k.perCtaHalf = w(i++);
            k.perCtaSize = w(i++);
            break;
        case 9: k.address32Half = w(i++); break;
        case 10: {
            uint32_t c = w(i++);
            for (uint32_t j = 0; j < c && i + 2 <= words; j++, i += 2) k.launchGrid.push_back({w(i), w(i + 1)});
            break;
        }
        default: return true; // unknown extension: stop, as the format says
        }
    }
    return true;
}

bool readObject(const std::vector<uint8_t>& obj, CompiledModule& out)
{
    auto buf = MemoryBuffer::getMemBuffer(StringRef((const char*)obj.data(), obj.size()), "agx.o", false);
    auto elfOr = object::ObjectFile::createELFObjectFile(buf->getMemBufferRef());
    if (!elfOr) {
        out.error = "cannot read the AGX object: " + toString(elfOr.takeError());
        return false;
    }
    auto* elf = dyn_cast<object::ELF64LEObjectFile>(elfOr->get());
    if (!elf) {
        out.error = "AGX object is not ELF64LE";
        return false;
    }
    StringRef text, records, globals;
    object::SectionRef textSec, globalsSec;
    for (const object::SectionRef& s : elf->sections()) {
        auto name = s.getName();
        if (!name) continue;
        auto contents = s.getContents();
        if (!contents) continue;
        if (*name == ".text") { text = *contents; textSec = s; }
        else if (*name == ".agx.kernels") records = *contents;
        else if (*name == ".agx.globals") { globals = *contents; globalsSec = s; out.globalsAlign = (uint32_t)std::max<uint64_t>(16, s.getAlignment().value()); }
    }
    // Kernel entry points: symbols in .text.
    struct Sym { uint64_t off, size; };
    std::map<std::string, Sym> syms;
    std::vector<uint64_t> starts;
    for (const object::SymbolRef& s : elf->symbols()) {
        auto sec = s.getSection();
        if (!sec || *sec == elf->section_end() || !textSec.getObject() || **sec != textSec) continue;
        auto name = s.getName();
        auto addr = s.getValue();
        if (!name || !addr || name->empty()) continue;
        object::ELFSymbolRef es(s);
        syms[name->str()] = {*addr, es.getSize()};
        starts.push_back(*addr);
    }
    std::sort(starts.begin(), starts.end());
    // Records.
    const uint8_t* rp = (const uint8_t*)records.data();
    size_t rn = records.size(), off = 0;
    while (off + 4 <= rn) {
        uint32_t sz = rd32(rp + off);
        if (sz < 12 || off + sz > rn) break;
        CompiledKernel k;
        if (!parseRecord(rp + off, sz, k)) {
            out.error = "malformed .agx.kernels record";
            return false;
        }
        auto it = syms.find(k.name);
        if (it == syms.end()) {
            out.error = "kernel " + k.name + " has no symbol in .text";
            return false;
        }
        uint64_t begin = it->second.off, end = text.size();
        auto nx = std::upper_bound(starts.begin(), starts.end(), begin);
        if (nx != starts.end()) end = *nx;
        if (it->second.size && begin + it->second.size <= text.size()) end = std::max<uint64_t>(begin + it->second.size, std::min<uint64_t>(end, text.size()));
        k.code.assign((const uint8_t*)text.data() + begin, (const uint8_t*)text.data() + end);
        out.kernels.push_back(std::move(k));
        off += sz;
    }
    out.globals.assign((const uint8_t*)globals.data(), (const uint8_t*)globals.data() + globals.size());
    if (!globals.empty()) {
        for (const object::SectionRef& s : elf->sections()) {
            auto target = s.getRelocatedSection();
            if (!target || *target == elf->section_end() || **target != globalsSec) continue;
            for (const object::RelocationRef& r : s.relocations()) {
                object::ELFRelocationRef er(r);
                auto addend = er.getAddend();
                auto sym = r.getSymbol();
                uint64_t symOff = 0;
                if (sym != elf->symbol_end()) {
                    auto v = sym->getValue();
                    if (v) symOff = *v;
                }
                out.relocs.push_back({r.getOffset(), (uint32_t)r.getType(), symOff, addend ? *addend : 0});
            }
        }
    }
    return true;
}

} // namespace

std::vector<std::string> kernelNames(const std::vector<uint8_t>& bitcode, std::string& error)
{
    LLVMContext ctx;
    auto buf = MemoryBuffer::getMemBuffer(StringRef((const char*)bitcode.data(), bitcode.size()), "air", false);
    auto mOr = parseBitcodeFile(buf->getMemBufferRef(), ctx, ParserCallbacks([](StringRef, StringRef) -> std::optional<std::string> {
                                    return std::nullopt;
                                }));
    std::vector<std::string> names;
    if (!mOr) {
        error = toString(mOr.takeError());
        return names;
    }
    for (auto& kv : readAIRKernels(**mOr)) names.push_back(kv.first);
    return names;
}

CompiledModule compileAIR(const std::vector<uint8_t>& bitcode, const CompileOptions& opts)
{
    auto t0 = std::chrono::steady_clock::now();
    CompiledModule out;
    initLLVM();
    LLVMContext ctx;
    std::string diag;
    ctx.setDiagnosticHandler(std::make_unique<DiagCollector>(&diag));
    auto buf = MemoryBuffer::getMemBuffer(StringRef((const char*)bitcode.data(), bitcode.size()), "air", false);
    // Keep AIR's data layout while reading: agx-lower-air switches it to AGX's.
    auto mOr = parseBitcodeFile(buf->getMemBufferRef(), ctx, ParserCallbacks([](StringRef, StringRef) -> std::optional<std::string> {
                                    return std::nullopt;
                                }));
    if (!mOr) {
        out.error = "cannot read AIR bitcode: " + toString(mOr.takeError());
        return out;
    }
    std::unique_ptr<Module> M = std::move(*mOr);
    if (!M->getNamedMetadata("air.kernel")) {
        out.error = "the AIR module has no !air.kernel (no kernels)";
        return out;
    }
    auto infos = readAIRKernels(*M);

    Triple triple("agx");
    std::string err;
    const Target* target = TargetRegistry::lookupTarget("", triple, err);
    if (!target) {
        out.error = "AGX target unavailable: " + err;
        return out;
    }
    TargetOptions to;
    CodeGenOptLevel ol = opts.optLevel >= 3 ? CodeGenOptLevel::Aggressive
                         : opts.optLevel == 2 ? CodeGenOptLevel::Default
                         : opts.optLevel == 1 ? CodeGenOptLevel::Less
                                              : CodeGenOptLevel::None;
    std::unique_ptr<TargetMachine> tm(target->createTargetMachine(triple, opts.cpu, opts.features, to, Reloc::Static, std::nullopt, ol));
    if (!tm) {
        out.error = "cannot create the AGX target machine for " + opts.cpu;
        return out;
    }
    M->setTargetTriple(triple);
    for (Function& F : *M) {
        if (F.isDeclaration()) continue;
        F.addFnAttr("target-cpu", opts.cpu);
        if (!opts.features.empty()) F.addFnAttr("target-features", opts.features);
    }
    if (verifyModule(*M, nullptr)) {
        std::string v;
        raw_string_ostream vs(v);
        verifyModule(*M, &vs);
        out.error = "AIR module does not verify: " + v;
        return out;
    }
    // agx-lower-air (the target's pipeline-start callback) and, unless
    // disabled, LLVM's optimization pipeline on the AGX module.
    {
        LoopAnalysisManager lam;
        FunctionAnalysisManager fam;
        CGSCCAnalysisManager cam;
        ModuleAnalysisManager mam;
        PassBuilder pb(ctx, tm.get());
        fam.registerPass([&] { return pb.buildDefaultAAPipeline(); });
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cam, mam);
        std::string pipeline = opts.irPipeline.empty() ? "agx-lower-air" : opts.irPipeline;
        ModulePassManager mpm;
        if (auto e = pb.parsePassPipeline(mpm, pipeline)) {
            out.error = "bad IR pipeline '" + pipeline + "': " + toString(std::move(e));
            return out;
        }
        mpm.run(*M, mam);
        if (!diag.empty()) {
            out.error = "AIR lowering failed:\n" + diag;
            return out;
        }
    }
    // Which lowered arguments are by-reference bytes (constant buffers held in uniforms).
    std::map<std::string, std::vector<bool>> byrefs;
    for (Function& F : *M) {
        if (F.isDeclaration() || F.getCallingConv() != CallingConv::AGX_KERNEL) continue;
        std::vector<bool> v;
        for (Argument& A : F.args()) {
            if (A.hasAttribute("agx-hidden")) continue;
            v.push_back(A.hasByRefAttr());
        }
        byrefs[F.getName().str()] = v;
    }
    SmallVector<char, 0> objBuf;
    {
        raw_svector_ostream os(objBuf);
        legacy::PassManager pm;
        if (tm->addPassesToEmitFile(pm, os, nullptr, CodeGenFileType::ObjectFile)) {
            out.error = "the AGX target cannot emit objects";
            return out;
        }
        pm.run(*M);
    }
    if (!diag.empty()) {
        out.error = "AGX compile failed:\n" + diag;
        return out;
    }
    out.object.assign(objBuf.begin(), objBuf.end());
    if (!readObject(out.object, out)) return out;
    for (auto& k : out.kernels) {
        auto it = infos.find(k.name);
        if (it == infos.end()) continue;
        const auto& bufs = it->second.buffers;
        for (size_t a = 0; a < k.args.size() && a < bufs.size(); a++) {
            k.args[a].location = bufs[a].first;
            k.args[a].airAddrSpace = bufs[a].second;
        }
        if (k.args.size() != bufs.size())
            fprintf(stderr, "metal-asahi: warning: kernel %s: %zu AGX arguments for %zu AIR buffers\n", k.name.c_str(),
                    k.args.size(), bufs.size());
        k.threadgroupBufferLocation = it->second.tgLocation;
        auto br = byrefs.find(k.name);
        if (br != byrefs.end())
            for (size_t a = 0; a < k.args.size() && a < br->second.size(); a++) k.args[a].byref = br->second[a];
    }
    out.compileSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

} // namespace masahi
