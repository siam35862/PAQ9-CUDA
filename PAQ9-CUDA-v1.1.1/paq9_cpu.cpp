// =====================================================================
// PAQ9-CPU — pure C++ port of PAQ9-CUDA
//
// WHAT CHANGED vs the CUDA warp-cooperative version:
//   1. GPU parallelism -> CPU parallelism. The unit of parallel work is
//      still one CHUNK, but a chunk is now processed entirely by a
//      single CPU thread (no warp, no lanes, no __shfl_sync/__syncwarp).
//      A small thread pool (size = hardware_concurrency(), capped at
//      num_of_chunks) pulls chunk indices from a shared atomic counter
//      and compresses/decompresses them independently.
//   2. All the warp-splitting that spread the 7 HashTable lookups and
//      11 StateMap updates across lanes 4-10 / 0-10 has been removed.
//      That splitting existed only to keep GPU lanes busy; on a CPU
//      thread the work is simply done sequentially, in the same order
//      lane 0 would have produced (so results are bit-identical to the
//      original algorithm; only the parallelization strategy differs).
//   3. The GPU "global object table" pattern (predictor[MAX_THREADS],
//      lzp[MAX_THREADS], indexed by chunk id, manually new'd/deleted)
//      is gone. Each worker thread constructs its own LZP/Predictor/
//      Encoder and all their backing buffers as plain local objects
//      (std::vector-owned), so they're automatically freed via RAII
//      when the thread finishes that chunk — no explicit delete[]s to
//      track, and no risk of one chunk's thread touching another
//      chunk's state.
//   4. cudaMallocManaged/cudaFree -> std::vector<T> (or new[]/delete[]
//      where a raw pointer is genuinely needed). No unified memory, no
//      device heap limit search — a chunk's working set is allocated
//      right before it's processed and released right after, so peak
//      host memory is roughly (thread pool size) x (per-chunk working
//      set), not (total chunk count) x (per-chunk working set) like the
//      original GPU version needed to fit within one device heap.
//   5. The device-call batching loop (splitting num_of_chunks into
//      several kernel launches to fit a GPU heap budget) is removed
//      entirely — the CPU thread pool just streams through all chunks.
//
// Everything else (Squash/Stretch/Ilog tables, State_table, StateMap,
// Mix, APM, HashTable, LZP, Predictor, Encoder bit-coding math, and the
// archive file format) is numerically identical to the original
// algorithm, so files produced by this program are byte-identical in
// content to a single-thread-per-chunk PAQ9 run (just produced by CPU
// threads pulling from a work queue instead of one GPU thread per
// chunk).
//
// Build:  g++ -O2 -std=c++17 -pthread paq9_cpu.cpp -o paq9_cpu
// Usage:  paq9_cpu -c <archive> [-<level 1-11>] <source_file>
//         paq9_cpu -d <archive> [<destination_file>]
// =====================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <memory>

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

#define COMPRESS 0
#define DECOMPRESS 1
#define endl std::endl
constexpr size_t MB = 1024 * 1024;

int memory_level = 1;       // default memory level MEM=1<<22+memory_level;
int memory_chunk_level = 1; // default memory chunk size, in MB
int level = 1;
size_t total_uncompressed_size = 0;
size_t total_compressed_size = 0;
unsigned long long total_heap_allocated = 0;

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
class Squash
{
    short tab[4096];

public:
    Squash();
    int operator()(int d) const;
};

Squash::Squash()
{
    static const int t[33] = {
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
        1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
        4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
    for (int i = -2048; i < 2048; ++i)
    {
        int w = i & 127;
        int d = (i >> 7) + 16;
        tab[i + 2048] = (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
    }
}
int Squash::operator()(int d) const
{
    d += 2048;
    if (d < 0)
        return 0;
    else if (d > 4095)
        return 4095;
    else
        return tab[d];
}

// global instance of squash — constructed once in main() before any
// worker thread is spawned, then only ever read, so sharing it across
// threads afterward is safe without further synchronization.
Squash *squash = nullptr;

//////////////////////////// Stretch ///////////////////////////////

class Stretch
{
    short t[4096];

public:
    explicit Stretch(const Squash &sq);
    int operator()(int p) const;
};

Stretch::Stretch(const Squash &sq)
{
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x)
    { // invert squash()
        int i = sq(x);
        for (int j = pi; j <= i; ++j)
            t[j] = x;
        pi = i + 1;
    }
    t[4095] = 2047;
}

int Stretch::operator()(int p) const
{
    assert(p >= 0 && p < 4096);
    return t[p];
}

Stretch *stretch = nullptr;

///////////////////////////// ilog //////////////////////////////

class Ilog
{
    U8 *table;

public:
    explicit Ilog(U8 *table);
    int operator()(U16 x) const;
    int operator()(U32 x) const;
};

Ilog::Ilog(U8 *table) : table(table)
{
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i)
    {
        x += 774541002 / (i * 2 - 1); // numerator is 2^29/ln 2
        table[i] = x >> 24;
    }
}
int Ilog::operator()(U16 x) const
{
    return table[x];
}
int Ilog::operator()(U32 x) const
{
    if (x >= 0x1000000)
        return 256 + table[x >> 16];
    else if (x >= 0x10000)
        return 128 + table[x >> 8];
    else
        return table[x];
}
Ilog *ilog = nullptr;
static U8 log_table[65536];

///////////////////////// state table ////////////////////////

static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

static U32 state_map_dt[1024];
static std::once_flag state_map_dt_once;

static void initStateMapTable()
{
    std::call_once(state_map_dt_once, []() {
        for (int i = 0; i < 1024; i++)
            state_map_dt[i] = 16384 / (i + i + 3);
    });
}

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;

public:
    explicit StateMap(U32 *prediction_table_ptr, int n = 256);
    void update(int y, int limit = 255);
    int predict_next_bit(int cntx);
};

StateMap::StateMap(U32 *prediction_table_ptr, int n) : N(n), cntxt(0), prediction_table(prediction_table_ptr)
{
    initStateMapTable();
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
}

void StateMap::update(int y, int limit)
{
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10;

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] = prediction_table[cntxt] & 0xfffffc00 | limit;

    prediction_table[cntxt] += (((y << 22) - p) >> 3) * state_map_dt[n] & 0xfffffc00;
}

int StateMap::predict_next_bit(int cntx)
{
    assert(cntx >= 0 && cntx < N);
    return prediction_table[cntxt = cntx] >> 20;
}

//////////////////////////// Mix, APM /////////////////////////

class Mix
{
protected:
    const int N;
    int *wt;
    int x1, x2;
    int context;
    int last_prediction;

public:
    explicit Mix(int *weight_ptr, int n = 512);
    int prediction(int p1, int p2, int cntxt);
    void update(int y);
};

Mix::Mix(int *weight_ptr, int n) : N(n), wt(weight_ptr), x1(0), x2(0), context(0), last_prediction(0)
{
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

int Mix::prediction(int p1, int p2, int cntxt)
{
    assert(cntxt >= 0 & cntxt < N);
    context = cntxt * 2;
    return last_prediction = ((x1 = p1) * (wt[context] >> 16) + (x2 = p2) * (wt[context + 1] >> 16) + 128) >> 8;
}

void Mix::update(int y)
{
    assert(y == 0 || y == 1);
    int error = ((y << 12) - squash->operator()(last_prediction));
    if ((wt[context] & 3) < 3)
    {
        error *= 4 - (++wt[context] & 3);
    }
    error = (error + 8) >> 4;
    wt[context] += x1 * error & -4;
    wt[context + 1] += x2 * error;
}

class APM : public Mix
{
public:
    APM(int *weight_ptr, int n);
};

APM::APM(int *weight_ptr, int n) : Mix(weight_ptr, n)
{
    for (int i = 0; i < n; i++)
    {
        wt[2 * i] = 0;
    }
}

//////////////////////////// HashTable /////////////////////////

template <int B>
class HashTable
{
    U8 *table;
    U8 *raw_table;
    const U32 N;

public:
    HashTable(int n, U8 *table_ptr);
    U8 *operator[](U32 i);
};

template <int B>
HashTable<B>::HashTable(int n, U8 *table_ptr) : table(table_ptr), raw_table(0), N(n)
{
    assert(B >= 2 && (B & B - 1) == 0);
    assert(N >= B * 4 && (N & N - 1) == 0);
    raw_table = table;
    table += 64 - int(reinterpret_cast<uintptr_t>(table) & 63);
}

template <int B>
U8 *HashTable<B>::operator[](U32 i)
{
    i *= 123456791;
    i = i << 16 | i >> 16;
    i *= 234567891;
    int chk = i >> 24;
    i = i * B & N - B;
    if (table[i] == chk)
        return table + i;
    if (table[i ^ B] == chk)
        return table + (i ^ B);
    if (table[i ^ B * 2] == chk)
        return table + (i ^ B * 2);
    if (table[i + 1] > table[i + 1 ^ B] || table[i + 1] > table[i + 1 ^ B * 2])
        i ^= B;

    if (table[i + 1] > table[i + 1 ^ B ^ B * 2])
        i ^= B ^ B * 2;
    memset(table + i, 0, B);
    table[i] = chk;
    return table + i;
}

////////////////////////// LZP /////////////////////////

static inline bool isalpha_host(char ch)
{
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z');
}

static inline char tolower_host(char ch)
{
    if (ch >= 'A' && ch <= 'Z')
        ch += 'a' - 'A';

    return ch;
}

class LZP
{
private:
    const size_t N, H;
    enum
    {
        MINLEN = 12
    };
    U8 *buffer;
    U32 *table;
    int match;
    size_t len;
    size_t pos;
    U32 hash;
    U32 hash1;
    U32 hash2;
    StateMap *statemap;
    APM *apm1, *apm2, *apm3;
    int literals, matches;

public:
    U32 word0, word1;
    // N.B. MEM here is the per-chunk memory budget (see chunkMEM()),
    // passed in explicitly instead of read from a __device__ global,
    // since each chunk/thread can in principle use its own budget.
    LZP(size_t MEM, StateMap *statemap1, U8 *buffer, U32 *table, APM *apm1, APM *apm2, APM *apm3);
    int predict_char();
    int context(int i);
    int context4() { return hash2; }
    int context8() { return hash1; }
    int probability();
    void update(int ch);
};

LZP::LZP(size_t MEM, StateMap *statemap, U8 *buf, U32 *tab, APM *apm1, APM *apm2, APM *apm3)
    : N(MEM / 8), H(MEM / 32),
      match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0),
      statemap(statemap), apm1(apm1), apm2(apm2), apm3(apm3),
      literals(0), matches(0), word0(0), word1(0)
{
    assert(MEM > 0);
    assert(H > 0);
    buffer = buf;
    table = tab;
}

int LZP::predict_char()
{
    return len >= MINLEN ? buffer[match & N - 1] : -1;
}

int LZP::context(int i)
{
    assert(i > 0);
    return buffer[pos - i & N - 1];
}

int LZP::probability()
{
    if (len < MINLEN)
        return 0;
    int cxt = static_cast<int>(len);
    if (len > 28)
        cxt = 28 + (len >= 32) + (len >= 64) + (len >= 128);
    int pc = predict_char();
    int pr = statemap->predict_next_bit(cxt);
    pr = stretch->operator()(pr);
    pr = apm1->prediction(2048, pr * 2, hash2 * 256 + pc & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(2048, pr * 2, hash1 * (11 << 6) + pc & 0x3ffff) * 3 + pr >> 2;
    pr = apm3->prediction(2048, pr * 2, hash1 * (7 << 4) + pc & 0xfffff) * 3 + pr >> 2;
    pr = squash->operator()(pr);
    return pr;
}

void LZP::update(int ch)
{
    int y = predict_char() == ch;
    hash1 = hash1 * (3 << 4) + ch + 1;
    hash2 = hash2 << 8 | ch;
    hash = hash * (5 << 2) + ch + 1 & H - 1;
    if (len >= MINLEN)
    {
        statemap->update(y);
        apm1->update(y);
        apm2->update(y);
        apm3->update(y);
    }
    if (isalpha_host(ch))
        word0 = word0 * (29 << 2) + tolower_host(ch);
    else if (word0)
        word1 = word0, word0 = 0;
    buffer[pos & N - 1] = ch;
    ++pos;
    if (y)
    {
        ++len;
        ++match;
        ++matches;
    }
    else
    {
        ++literals;
        y = 0;
        len = 1;
        match = table[hash];
        if (!((match ^ pos) & N - 1))
            --match;
        while (len <= 128 && buffer[match - len & N - 1] == buffer[pos - len & N - 1])
            ++len;
        --len;
    }
    table[hash] = pos;
}

//////////////////////////// Predictor /////////////////////////

class Predictor
{
    enum
    {
        N = 11
    };
    int c0;
    int nibble;
    int bcount;
    HashTable<16> *hashtable;
    StateMap *statemap[N];
    U8 *cp[N];
    U8 *sp[N];
    Mix *mix[N - 1];
    APM *apm1, *apm2, *apm3;
    U8 *context1;
    LZP *lzp; // this chunk's LZP instance (was a global array indexed by
              // chunk id on the GPU; now just a plain owned-elsewhere
              // pointer passed in at construction time)

public:
    Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp);
    // Sequential CPU version: does the same 7 HashTable lookups and 11
    // StateMap calls the GPU warp version spread across lanes, just one
    // after another. The Mix/APM chain was always serial (each stage
    // depends on the previous one), so that part is unchanged in spirit.
    int predict_next_bit();
    void update(int y);
};

Predictor::Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp)
    : c0(0), nibble(1), bcount(0), hashtable(hashtable_ptr),
      apm1(apm1), apm2(apm2), apm3(apm3), context1(context1_ptr), lzp(lzp)
{
    for (int i = 0; i < N; ++i)
    {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1)
            mix[i] = mix1[i];
    }
}

void Predictor::update(int y)
{
    assert(y == 0 || y == 1);

    if (c0 == 0)
    {
        c0 = 1 - y;
        return;
    }

    *sp[0] = nex(*sp[0], y);
    statemap[0]->update(y);
    for (int i = 1; i < N; ++i)
    {
        *sp[i] = nex(*sp[i], y);
        statemap[i]->update(y);
        mix[i - 1]->update(y);
    }

    c0 += c0 + y;
    bcount++;
    if (bcount == 8)
        bcount = c0 = 0;
    if ((nibble += nibble + y) >= 16)
        nibble = 1;
    apm1->update(y);
    apm2->update(y);
    apm3->update(y);
}

int Predictor::predict_next_bit()
{
    assert(lzp);
    if (c0 == 0)
        return lzp->probability();

    int pc = lzp->predict_char();
    int r = pc + 256 >> 8 - bcount == c0;
    U32 c4 = lzp->context4();
    U32 c8 = (lzp->context8() << 4) - 1;

    if ((bcount & 3) == 0)
    { // nibble boundary? update context pointers
        int pcr = pc & -r;
        U32 c4p = c4 << 8;

        if (bcount == 0)
        { // byte boundary? update order-1 context pointers
            cp[0] = context1 + (c4 >> 16 & 0xff00);
            cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
            cp[2] = context1 + (c4 & 0xff00) + 0x20000;
            cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
        }

        // 7 heavy HashTable lookups — done sequentially on this thread
        // (on the GPU version these were spread across lanes 4-10).
        cp[4] = hashtable->operator[]((c4p & 0xffff00) - c0);
        cp[5] = hashtable->operator[]((c4p & 0xffffff00) * 3 + c0);
        cp[6] = hashtable->operator[](c4 * 7 + c0);
        cp[7] = hashtable->operator[]((c8 * 5 & 0xfffffc) + c0);
        cp[8] = hashtable->operator[]((c8 * 11 & 0xffffff0) + c0 + pcr * 13);
        cp[9] = hashtable->operator[]((lzp->word0 * 5 + c0 + pcr * 17));
        cp[10] = hashtable->operator[]((lzp->word1 * 7 + lzp->word0 * 11 + c0 + pcr * 37));
    }

    r <<= 8;

    // 11 StateMap predict_next_bit() calls, then the serial Mix/APM chain.
    sp[0] = &cp[0][c0];
    int pr = stretch->operator()(statemap[0]->predict_next_bit(*sp[0]));
    for (int i = 1; i < N; ++i)
    {
        sp[i] = &cp[i][i < 4 ? c0 : nibble];
        int st_i = *sp[i];
        int stretched_i = stretch->operator()(statemap[i]->predict_next_bit(st_i));
        pr = mix[i - 1]->prediction(pr, stretched_i, st_i + r) * 3 + pr >> 2;
    }
    pr = apm1->prediction(512, pr * 2, c0 + pc * 256 & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(512, pr * 2, c4 << 8 & 0xff00 | c0) * 3 + pr >> 2;
    pr = apm3->prediction(512, pr * 2, c4 * 3 + c0 & 0xffff) * 3 + pr >> 2;
    pr = squash->operator()(pr);
    return pr;
}

//////////////////////////// Encoder ////////////////////////////

class Encoder
{
private:
    const int mode;
    char *inout;
    size_t total_size;
    Predictor *predictor;

    U32 x1, x2;
    U32 x;
    enum
    {
        BUFSIZE = 0x20000
    };
    U8 *buffer;
    size_t usize, csize;

public:
    size_t iterator_size;
    Encoder(int m, char *temp, unsigned char *buffer_ptr, Predictor *predictor, size_t tsz, size_t itr);
    bool flush();
    bool put4(U32 c);
    int code(int y = 0);
    bool count();
};

Encoder::Encoder(int m, char *temp, unsigned char *buffer_ptr, Predictor *predictor, size_t tsz, size_t itr)
    : mode(m), inout(temp), total_size(tsz), predictor(predictor),
      x1(0), x2(0xffffffff), x(0), buffer(buffer_ptr),
      usize(0), csize(0), iterator_size(itr)
{
    if (mode == DECOMPRESS)
    { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (inout[iterator_size++] & 255);
        csize = 4;
    }
}

bool Encoder::put4(U32 c)
{
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 24);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 16);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c >> 8);
    if (iterator_size > total_size)
        return false;
    inout[iterator_size++] = char(c);
    return true;
}

bool Encoder::flush()
{
    if (mode == COMPRESS)
    {
        buffer[csize++] = x1 >> 24;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        if (!put4((U32)usize))
            return false;
        if (!put4((U32)csize))
            return false;
        for (size_t i = 0; i < csize; i++)
        {
            if (iterator_size > total_size)
                return false;
            inout[iterator_size++] = buffer[i];
        }
        x1 = x = 0;
        usize = csize = 0;
        x2 = 0xffffffff;
        return true;
    }
    return true;
}

int Encoder::code(int y)
{
    int p = predictor->predict_next_bit();
    assert(p >= 0 && p < 4096);
    p += p < 2048;

    U32 xmid = x1 + (x2 - x1 >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
    assert(xmid >= x1 && xmid < x2);
    if (mode == DECOMPRESS)
        y = x <= xmid;
    y ? (x2 = xmid) : (x1 = xmid + 1);

    predictor->update(y);

    while (((x1 ^ x2) & 0xff000000) == 0)
    { // pass equal leading bytes of range
        if (mode == COMPRESS)
            buffer[csize++] = x2 >> 24;
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
        if (mode == DECOMPRESS)
            x = (x << 8) + (inout[iterator_size++] & 255);
    }
    return y;
}

bool Encoder::count()
{
    assert(mode == COMPRESS);
    ++usize;
    if (csize > BUFSIZE - 256)
        return flush();
    return true;
}

static U32 get4(size_t &itr, const char *in)
{
    U32 r = (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    return r;
}

// =====================================================================
// Per-chunk working set. Replaces the CUDA ThreadBuffers + per-thread
// cudaMallocTracked() calls: each chunk gets its own scratch buffers,
// sized from that chunk's memory_level, allocated right before the
// chunk is processed and released automatically (vector dtors) right
// after — so peak memory tracks (thread pool size), not (chunk count).
// =====================================================================
struct ChunkBuffers
{
    std::vector<U32> lzp_statemap;
    std::vector<int> lzp_apm[3];
    std::vector<U8> lzp_buffer;
    std::vector<U32> lzp_table;

    std::vector<U32> predictor_statemap[11];
    std::vector<int> predictor_mix[10];
    std::vector<int> predictor_apm[3];
    std::vector<U8> predictor_hashtable;
    std::vector<U8> predictor_context1;
    std::vector<U8> encoder_buffer;

    explicit ChunkBuffers(int mem_level)
    {
        size_t MEM_host = 1ull << (22 + mem_level);

        lzp_statemap.resize(512);
        lzp_apm[0].resize(131072);
        lzp_apm[1].resize(0x80000);
        lzp_apm[2].resize(0x200000);
        lzp_buffer.resize(MEM_host / 8);
        lzp_table.resize(MEM_host / 32);

        for (int j = 0; j < 11; j++)
            predictor_statemap[j].resize(0x100);
        for (int j = 0; j < 10; j++)
            predictor_mix[j].resize(0x800);
        predictor_apm[0].resize(0x20000);
        predictor_apm[1].resize(0x20000);
        predictor_apm[2].resize(0x20000);
        predictor_hashtable.resize(MEM_host / 2 + 128);
        predictor_context1.resize(0x40000);
        encoder_buffer.resize(0x20000);
    }
};

// Builds LZP + Predictor (and all their backing StateMap/Mix/APM/
// HashTable objects) for one chunk, given that chunk's ChunkBuffers.
// Everything returned is owned by the `holders` vectors passed in by
// the caller (kept alive for the duration of chunk processing) so the
// caller can just let them go out of scope when done — no manual
// delete, unlike the GPU version's Predictor/LZP destructors.
struct ChunkModel
{
    std::vector<std::unique_ptr<StateMap>> smOwn;
    std::vector<std::unique_ptr<Mix>> mixOwn;
    std::unique_ptr<APM> lzpApm1, lzpApm2, lzpApm3;
    std::unique_ptr<APM> predApm1, predApm2, predApm3;
    std::unique_ptr<HashTable<16>> hashtable;
    std::unique_ptr<StateMap> lzpSM;
    std::unique_ptr<LZP> lzp;
    std::unique_ptr<Predictor> predictor;
};

static void buildChunkModel(ChunkBuffers &cb, size_t chunkMEM, ChunkModel &m)
{
    m.lzpSM = std::make_unique<StateMap>(cb.lzp_statemap.data(), 0x200);
    m.lzpApm1 = std::make_unique<APM>(cb.lzp_apm[0].data(), 0x10000);
    m.lzpApm2 = std::make_unique<APM>(cb.lzp_apm[1].data(), 0x40000);
    m.lzpApm3 = std::make_unique<APM>(cb.lzp_apm[2].data(), 0x100000);
    m.lzp = std::make_unique<LZP>(chunkMEM, m.lzpSM.get(), cb.lzp_buffer.data(), cb.lzp_table.data(),
                                   m.lzpApm1.get(), m.lzpApm2.get(), m.lzpApm3.get());

    StateMap *psm[11];
    for (int j = 0; j < 11; ++j)
    {
        m.smOwn.push_back(std::make_unique<StateMap>(cb.predictor_statemap[j].data(), 0x100));
        psm[j] = m.smOwn.back().get();
    }
    Mix *pmix[10];
    for (int j = 0; j < 10; ++j)
    {
        m.mixOwn.push_back(std::make_unique<Mix>(cb.predictor_mix[j].data(), 0x400));
        pmix[j] = m.mixOwn.back().get();
    }
    m.predApm1 = std::make_unique<APM>(cb.predictor_apm[0].data(), 0x10000);
    m.predApm2 = std::make_unique<APM>(cb.predictor_apm[1].data(), 0x10000);
    m.predApm3 = std::make_unique<APM>(cb.predictor_apm[2].data(), 0x10000);
    m.hashtable = std::make_unique<HashTable<16>>((int)(chunkMEM / 2), cb.predictor_hashtable.data());

    m.predictor = std::make_unique<Predictor>(cb.predictor_context1.data(), psm, pmix,
                                               m.predApm1.get(), m.predApm2.get(), m.predApm3.get(),
                                               m.hashtable.get(), m.lzp.get());
}

static size_t chunkMEM(int mem_level)
{
    return (size_t)1 << (19 + mem_level);
}

// =====================================================================
// One chunk, compress. Runs entirely on a single worker thread — this
// is the direct CPU analogue of the old paq9_cuda kernel body's
// COMPRESS branch, minus all warp bookkeeping.
// =====================================================================
static void compressChunk(const char *input, size_t inputSize, int mem_level,
                           std::vector<char> &output, size_t &outputSize)
{
    ChunkBuffers cb(mem_level);
    ChunkModel model;
    buildChunkModel(cb, chunkMEM(mem_level), model);
    LZP &lzp = *model.lzp;
    Predictor &predictor = *model.predictor;

    output.assign(inputSize + 2, 0); // +2 mirrors the host-side (chunk_B+2) slack
    Encoder encoder(COMPRESS, output.data(), cb.encoder_buffer.data(), &predictor, inputSize, 0);

    output[encoder.iterator_size++] = '0';

    bool store_mode = false;
    for (size_t i = 0; i < inputSize; ++i)
    {
        int ch = (unsigned char)input[i];
        int cp = lzp.predict_char();
        if (ch == cp)
        {
            encoder.code(1);
        }
        else
        {
            for (int b = 8; b >= 0; --b)
                encoder.code(ch >> b & 1);
        }

        if (!encoder.count())
        {
            store_mode = true;
            break;
        }
        lzp.update(ch);
    }

    if (!store_mode)
    {
        if (!encoder.flush())
            store_mode = true;
    }

    if (store_mode)
    {
        output.assign(inputSize + 1, 0);
        output[0] = '1';
        if (inputSize)
            std::memcpy(output.data() + 1, input, inputSize);
        outputSize = inputSize + 1;
    }
    else
    {
        outputSize = encoder.iterator_size;
    }
}

// =====================================================================
// One chunk, decompress.
// =====================================================================
static void decompressChunk(const char *input, size_t inputSize, int mem_level,
                             size_t expectedOutputSize,
                             std::vector<char> &output, size_t &outputSize)
{
    if (inputSize == 0)
    {
        outputSize = 0;
        return;
    }
    if (input[0] == '1')
    {
        output.assign(input + 1, input + inputSize);
        outputSize = output.size();
        return;
    }

    ChunkBuffers cb(mem_level);
    ChunkModel model;
    buildChunkModel(cb, chunkMEM(mem_level), model);
    LZP &lzp = *model.lzp;
    Predictor &predictor = *model.predictor;

    output.assign(expectedOutputSize, 0);
    size_t outPos = 0;
    size_t itr = 1; // skip leading '0'

    // input is treated as non-const scratch by Encoder only in COMPRESS
    // mode; in DECOMPRESS mode it only reads from it, so this cast is safe.
    char *inMutable = const_cast<char *>(input);

    while (itr < inputSize)
    {
        size_t usize = get4(itr, input);
        get4(itr, input); // csize, discarded (as in the original)

        Encoder encoder(DECOMPRESS, inMutable, cb.encoder_buffer.data(), &predictor, inputSize, itr);

        for (size_t k = 0; k < usize; ++k)
        {
            int cp = lzp.predict_char();
            int first = encoder.code();
            if (first == 0)
            {
                cp = 1;
                while (cp < 256)
                    cp += cp + encoder.code();
                cp &= 255;
            }
            if (outPos < output.size())
                output[outPos++] = (char)cp;
            else
                output.push_back((char)cp), ++outPos;
            lzp.update(cp);
        }

        itr = encoder.iterator_size;
    }

    outputSize = outPos;
}

//////////////////////// file-format helpers ////////////////////////

void put4_stream(U32 c, std::ostream &out)
{
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}
U32 get4_stream(std::istream &in)
{
    U32 r = in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    return r;
}
void put8_stream(size_t c, std::ostream &out)
{
    out.put((c >> 56) & 0xFF);
    out.put((c >> 48) & 0xFF);
    out.put((c >> 40) & 0xFF);
    out.put((c >> 32) & 0xFF);
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}
size_t get8_stream(std::istream &in)
{
    size_t r = 0;
    for (int i = 0; i < 8; i++)
        r = r * 256 + in.get();
    return r;
}

size_t getBytesFromMemoryLevel(int n)
{
    if (n < 1 || n > 9)
        return SIZE_MAX;
    size_t MEM = (size_t)1 << (19 + n);
    return (size_t)(0.75 * MEM) + 13030528ULL;
}
int getMemoryLevelFromBytes(size_t bytes)
{
    for (int n = 1; n <= 9; ++n)
        if (bytes <= getBytesFromMemoryLevel(n))
            return n;
    return 9;
}

bool check_archive(std::istream &in)
{
    std::string magic = "PAQ9-CUDA"; // archive tag kept for format compatibility
    for (char c : magic)
        if (in.get() != c)
            return false;
    return in.get() == 1;
}
std::string get_file_name(std::istream &in)
{
    std::string file_name;
    char c;
    while (in.get(c) && c != '\0')
        file_name += c;
    return file_name;
}

// =====================================================================
// unsigned int hardwareThreads() — size of the CPU worker pool.
// =====================================================================
static unsigned int hardwareThreads(int cappedBy)
{
    unsigned int n = std::thread::hardware_concurrency();
    if (n == 0)
        n = 1;
    if (cappedBy > 0)
        n = std::min(n, (unsigned int)cappedBy);
    return std::max(1u, n);
}

// =====================================================================
// compress() — reads the whole source file into memory, splits it into
// fixed-size chunks, and runs a CPU thread pool over the chunk indices
// (an atomic counter is the work queue). Each worker calls
// compressChunk() for whatever index it claims; results are collected
// into per-chunk buffers and written to the archive in chunk order
// once every thread has finished.
// =====================================================================
void compress(const char *destination_file, const char *source_file)
{
    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_B = (size_t)source.tellg();
    source.seekg(0, std::ios::beg);

    std::vector<char> fileData(total_B);
    if (total_B)
        source.read(fileData.data(), total_B);
    source.close();

    memory_chunk_level = (1 << (level - 1));
    memory_level = getMemoryLevelFromBytes((size_t)memory_chunk_level * MB);
    size_t chunk_B = (size_t)memory_chunk_level * MB;
    int num_of_chunks = total_B == 0 ? 0 : (int)((total_B + chunk_B - 1) / chunk_B);

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest)
    {
        std::cout << std::string(destination_file) << " does not created/opened.\n";
        exit(1);
    }

    dest.write("PAQ9-CUDA", 9);
    dest.put(1);
    dest.write(source_file, strlen(source_file));
    dest.put(0);
    dest.put('c');
    put8_stream(total_B, dest);
    put4_stream(memory_chunk_level, dest);
    put4_stream(memory_level, dest);
    put4_stream(level, dest);
    put4_stream(num_of_chunks, dest);

    std::cout << "Memory Chunk Level: " << memory_chunk_level << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;

    std::vector<std::vector<char>> outputs(num_of_chunks);
    std::vector<size_t> outputSizes(num_of_chunks, 0);
    std::atomic<int> nextChunk(0);

    unsigned int numThreads = hardwareThreads(num_of_chunks);
    std::cout << "CPU worker threads: " << numThreads << " for " << num_of_chunks << " chunks" << endl;

    auto worker = [&]() {
        while (true)
        {
            int idx = nextChunk.fetch_add(1);
            if (idx >= num_of_chunks)
                break;
            size_t offset = (size_t)idx * chunk_B;
            size_t curSize = std::min(chunk_B, total_B - offset);
            compressChunk(fileData.data() + offset, curSize, memory_level, outputs[idx], outputSizes[idx]);
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> pool;
    for (unsigned int i = 0; i < numThreads; ++i)
        pool.emplace_back(worker);
    for (auto &t : pool)
        t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Compression time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << " ms" << endl;

    total_uncompressed_size = 0;
    total_compressed_size = 0;
    for (int i = 0; i < num_of_chunks; ++i)
    {
        size_t offset = (size_t)i * chunk_B;
        size_t curSize = std::min(chunk_B, total_B - offset);
        put4_stream((U32)curSize, dest);
        put4_stream((U32)outputSizes[i], dest);
        dest.write(outputs[i].data(), outputSizes[i]);
        total_uncompressed_size += curSize;
        total_compressed_size += outputSizes[i];
    }
    dest.close();

    std::cout << "Total: " << total_uncompressed_size << " Byte -> " << total_compressed_size << " Byte" << endl;
    if (total_compressed_size)
        std::cout << "Compression Ratio: " << 1.0 * total_uncompressed_size / total_compressed_size << endl;
}

// =====================================================================
// decompress() — parses the archive header, does a quick sequential
// pass over the chunk table to record each chunk's byte offset/size
// (needed because chunk lengths vary, so offsets can't be computed
// without reading the headers), then runs the same kind of CPU thread
// pool as compress() to decode chunks in parallel.
// =====================================================================
void decompress(const char *destination_file, const char *source_file)
{
    total_compressed_size = 0;
    total_uncompressed_size = 0;

    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << endl;
        exit(1);
    }

    if (!check_archive(source))
    {
        std::cout << "This is not a PAQ9-CUDA compressed file.\n";
        exit(1);
    }
    std::string filename = get_file_name(source);
    if (destination_file == nullptr)
        destination_file = filename.c_str();

    char mode = source.get();
    if (mode != 'c')
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }

    /*size_t usize_total =*/ get8_stream(source);
    memory_chunk_level = get4_stream(source);
    memory_level = get4_stream(source);
    level = get4_stream(source);
    int num_of_chunks = (int)get4_stream(source);

    std::cout << "Memory Chunk Level: " << memory_chunk_level << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;

    std::streampos bodyStart = source.tellg();
    source.seekg(0, std::ios::end);
    size_t bodyLen = (size_t)source.tellg() - (size_t)bodyStart;
    source.seekg(bodyStart);
    std::vector<char> body(bodyLen);
    if (bodyLen)
        source.read(body.data(), bodyLen);
    source.close();

    struct ChunkInfo
    {
        size_t uncompressed;
        size_t offset;
        size_t compressedSize;
    };
    std::vector<ChunkInfo> chunkInfo(num_of_chunks);
    size_t pos = 0;
    for (int i = 0; i < num_of_chunks; ++i)
    {
        size_t p = pos;
        U32 usize = get4(p, body.data());
        U32 csize = get4(p, body.data());
        chunkInfo[i] = {usize, p, csize};
        pos = p + csize;
    }

    std::vector<std::vector<char>> outputs(num_of_chunks);
    std::vector<size_t> outSizes(num_of_chunks, 0);
    std::atomic<int> nextChunk(0);

    unsigned int numThreads = hardwareThreads(num_of_chunks);
    std::cout << "CPU worker threads: " << numThreads << " for " << num_of_chunks << " chunks" << endl;

    auto worker = [&]() {
        while (true)
        {
            int idx = nextChunk.fetch_add(1);
            if (idx >= num_of_chunks)
                break;
            const ChunkInfo &ci = chunkInfo[idx];
            decompressChunk(body.data() + ci.offset, ci.compressedSize, memory_level,
                             ci.uncompressed, outputs[idx], outSizes[idx]);
        }
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> pool;
    for (unsigned int i = 0; i < numThreads; ++i)
        pool.emplace_back(worker);
    for (auto &t : pool)
        t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Decompression time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
              << " ms" << endl;

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest)
    {
        std::cout << std::string(destination_file) << " does not created/opened.\n";
        exit(1);
    }
    for (int i = 0; i < num_of_chunks; ++i)
    {
        dest.write(outputs[i].data(), outSizes[i]);
        total_uncompressed_size += outSizes[i];
        total_compressed_size += chunkInfo[i].compressedSize;
    }
    dest.close();

    std::cout << "Total: " << total_compressed_size << " Byte -> " << total_uncompressed_size << " Byte" << endl;
}

int main(int argc, char **args)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "CPU (multi-threaded) version of PAQ9 started successfully.\n\n";

    // one-time global table setup, done before any worker thread exists
    static Squash squashObj;
    static Stretch stretchObj(squashObj);
    static Ilog ilogObj(log_table);
    squash = &squashObj;
    stretch = &stretchObj;
    ilog = &ilogObj;
    initStateMapTable();

    if (argc < 3)
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }
    int mode;
    char *destination_file_name = 0;
    char *source_file_name = 0;
    if (args[1][0] == '-')
    {
        if (args[1][1] == 'c')
            mode = COMPRESS;
        else if (args[1][1] == 'd')
            mode = DECOMPRESS;
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }
    }
    else
    {
        std::cout << "Run again and provide arguments in correct way.\n";
        exit(1);
    }
    std::cout << "Working mode: " << (mode == COMPRESS ? "Compressing" : "Decompressing") << endl;

    int ind = 2;
    if (mode == COMPRESS)
    {
        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }

        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;
            size_t len = strlen(args[ind]);
            for (size_t i = 1; i < len; i++)
            {
                if (isdigit(args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    std::cout << "Run again and provide arguments in correct way.\n";
                    exit(1);
                }
            }
            level = stoi(temp);
            if (level < 1 || level > 11)
            {
                level = 1;
                std::cout << "Your provided level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc)
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }

        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }

        compress(destination_file_name, source_file_name);
    }
    else
    {
        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            std::cout << "Run again and provide arguments in correct way.\n";
            exit(1);
        }
        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }

        decompress(destination_file_name, source_file_name);
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Time Taken: " << seconds << " seconds\n";
    if (seconds > 0)
        std::cout << "Compression/Decompression Speed: "
                  << (mode == COMPRESS ? total_uncompressed_size : total_uncompressed_size) / seconds / 1024
                  << " KB/seconds\n";

    return 0;
}
