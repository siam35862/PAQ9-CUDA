// =====================================================================
// PAQ9-CPU -- CPU multi-threaded port of PAQ9-CUDA (warp-cooperative
// variant).
//
// WHAT CHANGED vs the CUDA source:
//   - Every __device__ class (Squash, Stretch, Ilog, StateMap, Mix, APM,
//     HashTable<B>, LZP, Predictor, Encoder) is now plain host C++. The
//     per-bit math inside Predictor::predict_next_bit()/update() and
//     Encoder::code() is copied field-for-field from the CUDA source's
//     lane-0 (serial-owner) path -- the warp-cooperative version spreads
//     11 independent StateMap lookups and 7 independent HashTable
//     lookups across 32 CUDA lanes purely for GPU throughput; it does
//     not change what gets computed. Executing that same sequence of
//     operations on one CPU thread, in the same order, produces bit-
//     identical results.
//   - Parallelism moves from "32 GPU lanes per chunk, many chunks per
//     kernel launch" to "one CPU thread per chunk, N chunks in flight at
//     once" (N = hardware_concurrency by default). Chunks are still
//     compressed/decompressed completely independently of each other
//     (this was already true on the GPU: predictor[]/lzp[] were indexed
//     per chunk), so no cross-thread synchronization is needed beyond a
//     work-stealing counter and a join at the end of each batch.
//   - The on-disk archive format (magic "PAQ9-CUDA", header fields,
//     per-chunk [input_size][output_size][data] records, and the
//     per-chunk '0'=coded / '1'=stored marker byte) is kept byte-for-
//     byte identical on purpose: a file compressed by the original
//     GPU build and a file compressed by this CPU build decompress to
//     the same output on EITHER implementation, and vice versa, as
//     long as they agree on -<memory_level> and -<chunk_level>. Those
//     two knobs size the model tables and chunk length and must match
//     between the compressor and decompressor, exactly as before.
//
// Always verify with a full compress -> decompress -> byte-diff round
// trip after building, starting with small inputs, before trusting it
// on real data -- same caveat as the CUDA source, this is still a
// correctness-sensitive adaptive arithmetic coder.
// =====================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cassert>
#include <cstdint>
#include <cctype>
#include <thread>
#include <atomic>
#include <memory>
#include <algorithm>

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

#define COMPRESS 0
#define DECOMPRESS 1
#define endl std::endl
constexpr size_t MB = 1024 * 1024;
#define base_memory_level 19 // default base memory level, MEM=1<<base_memory_level+memory_level
int memory_level = 1;        // default memory level MEM=1<<base_memory_level+memory_level;
int chunk_MB = 1;            // default memory chunks 1MB
int chunk_level = 1;
size_t total_uncompressed_size = 0;
size_t total_compressed_size = 0;

// Set once, before any worker thread is started, from -<memory_level>.
// Read-only afterwards, so sharing it across threads is safe.
size_t MEM = 1ULL << (base_memory_level + 1);

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

// global instance of squash
Squash *squash = nullptr;

//////////////////////////// Stretch ///////////////////////////////

class Stretch
{
    short t[4096];

public:
    Stretch();
    int operator()(int p) const;
};

Stretch::Stretch()
{
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x)
    { // invert squash()
        int i = squash->operator()(x);
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
// Built for parity with the CUDA source; not otherwise referenced there
// either -- kept so both sides stay structurally identical.

class Ilog
{
    U8 *table;

public:
    Ilog(U8 *table);
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
U8 log_table[0x10000];

///////////////////////// state table ////////////////////////

static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

int state_map_dt[1024];

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;

public:
    StateMap(U32 *prediction_table_ptr, int n = 256);
    ~StateMap();
    void update(int y, int limit = 255);
    int predict_next_bit(int cntx);
};

StateMap::StateMap(U32 *prediction_table_ptr, int n) : N(n), cntxt(0), prediction_table(prediction_table_ptr)
{
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
    if (state_map_dt[0] == 0)
        for (int i = 0; i < 1024; i++)
            state_map_dt[i] = 16384 / (i + i + 3);
}

StateMap::~StateMap()
{
    prediction_table = 0;
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
    Mix(int *weight_ptr, int n = 512);
    ~Mix();
    int prediction(int p1, int p2, int cntxt);
    void update(int y);
};

Mix::Mix(int *weight_ptr, int n) : N(n), wt(weight_ptr), x1(0), x2(0), context(0), last_prediction(0)
{
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

Mix::~Mix()
{
    wt = 0;
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
    HashTable(U32 n, U8 *table_ptr);
    ~HashTable();
    U8 *operator[](U32 i);
};

template <int B>
HashTable<B>::HashTable(U32 n, U8 *table_ptr) : table(table_ptr), raw_table(0), N(n)
{
    assert(B >= 2 && (B & B - 1) == 0);
    assert(N >= (U32)(B * 4) && (N & N - 1) == 0);
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

template <int B>
HashTable<B>::~HashTable()
{
    raw_table = table = 0;
}

////////////////////////// LZP /////////////////////////

inline bool isalpha_host(char ch)
{
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z');
}

inline char tolower_host(char ch)
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
    LZP(StateMap *statemap1, U8 *buffer, U32 *table, APM *apm1, APM *apm2, APM *apm3);
    ~LZP();
    int predict_char();
    int context(int i);
    int context4()
    {
        return hash2;
    }
    int context8()
    {
        return hash1;
    }
    int probability();
    void update(int ch);
};

LZP::LZP(StateMap *statemap, U8 *buf, U32 *tab, APM *apm1, APM *apm2, APM *apm3) : N(MEM / 8), H(MEM / 32),
                                                                                    match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0),
                                                                                    statemap(statemap), apm1(apm1), apm2(apm2), apm3(apm3),
                                                                                    literals(0), matches(0), word0(0), word1(0)
{
    assert(MEM > 0);
    assert(H > 0);
    buffer = buf;
    table = tab;
}

LZP::~LZP()
{
    delete statemap;
    delete apm1;
    delete apm2;
    delete apm3;
    table = 0;
    buffer = 0;
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
//
// The CUDA warp-cooperative version spreads the 11 StateMap lookups and
// the 7 heavy HashTable lookups below across lanes 0-10 and 4-10 of a
// warp so they execute concurrently on the GPU; the Mix/APM chain stays
// serial there too (each stage depends on the previous one). On a CPU
// thread there is only one "lane", so predict_next_bit()/update() below
// just perform lane 0's steps, then lanes 1..10's steps, in the same
// order the warp version issues them -- same reads, same writes, same
// result.

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
    LZP *lzp_ref; // this chunk's LZP object (was a global lzp[chunk] lookup in the CUDA source)

public:
    Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp_ref);
    ~Predictor();
    int predict_next_bit();
    void update(int y);
};

Predictor::Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr, LZP *lzp_ref) : c0(0), nibble(1), bcount(0),
                                                                                                                                                                   hashtable(hashtable_ptr), apm1(apm1), apm2(apm2), apm3(apm3), context1(context1_ptr), lzp_ref(lzp_ref)
{
    for (int i = 0; i < N; ++i)
    {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1)
            mix[i] = mix1[i];
    }
}

Predictor::~Predictor()
{
    for (int i = 0; i < N; ++i)
    {
        delete statemap[i];
        if (i < N - 1)
            delete mix[i];
    }
    delete apm1;
    delete apm2;
    delete apm3;
    delete hashtable;
    context1 = 0;
}

// Update model. Lane 0's part (c0==0 special case, statemap[0]/sp[0],
// c0/bcount/nibble bookkeeping, apm1/2/3) followed by lanes 1..10's part
// (statemap[i]/sp[i]/mix[i-1]) -- independent of lane 0's part in the
// original, so running them one after another on a single thread gives
// the same end state.
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
    for (int lane = 1; lane < N; ++lane)
    {
        *sp[lane] = nex(*sp[lane], y);
        statemap[lane]->update(y);
        mix[lane - 1]->update(y);
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

// Predict next bit -- same sequence of reads/writes as the CUDA warp
// version's lane 0 (scalar setup), lanes 4-10 (7 independent HashTable
// lookups), lanes 0-10 (11 independent StateMap lookups), then lane 0's
// serial Mix/APM chain.
int Predictor::predict_next_bit()
{
    assert(lzp_ref);
    if (c0 == 0)
    {
        return lzp_ref->probability();
    }

    int pc = lzp_ref->predict_char();
    int r = pc + 256 >> 8 - bcount == c0;
    U32 c4 = lzp_ref->context4();
    U32 c8 = (lzp_ref->context8() << 4) - 1;
    int bc = bcount;

    if ((bc & 3) == 0)
    { // nibble boundary? update context pointers
        int pcr = pc & -r;
        U32 c4p = c4 << 8;

        if (bc == 0)
        { // byte boundary? update order-1 context pointers
            cp[0] = context1 + (c4 >> 16 & 0xff00);
            cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
            cp[2] = context1 + (c4 & 0xff00) + 0x20000;
            cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
        }

        // 7 heavy HashTable lookups
        cp[4] = hashtable->operator[]((c4p & 0xffff00) - c0);
        cp[5] = hashtable->operator[]((c4p & 0xffffff00) * 3 + c0);
        cp[6] = hashtable->operator[](c4 * 7 + c0);
        cp[7] = hashtable->operator[]((c8 * 5 & 0xfffffc) + c0);
        cp[8] = hashtable->operator[]((c8 * 11 & 0xffffff0) + c0 + pcr * 13);
        cp[9] = hashtable->operator[]((lzp_ref->word0 * 5 + c0 + pcr * 17));
        cp[10] = hashtable->operator[]((lzp_ref->word1 * 7 + lzp_ref->word0 * 11 + c0 + pcr * 37));
    }

    // 11 StateMap predict_next_bit() calls
    r <<= 8;
    int stretched_cache[N];
    sp[0] = &cp[0][c0];
    stretched_cache[0] = stretch->operator()(statemap[0]->predict_next_bit(*sp[0]));
    for (int lane = 1; lane < N; ++lane)
    {
        sp[lane] = &cp[lane][lane < 4 ? c0 : nibble];
        int st = *sp[lane];
        stretched_cache[lane] = stretch->operator()(statemap[lane]->predict_next_bit(st));
    }

    // serial Mix + APM chain
    int pr = stretched_cache[0];
    for (int i = 1; i < N; ++i)
    {
        int st_i = *sp[i];
        int stretched_i = stretched_cache[i];
        pr = mix[i - 1]->prediction(pr, stretched_i, st_i + r) * 3 + pr >> 2;
    }
    pr = apm1->prediction(512, pr * 2, c0 + pc * 256 & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(512, pr * 2, c4 << 8 & 0xff00 | c0) * 3 + pr >> 2;
    pr = apm3->prediction(512, pr * 2, c4 * 3 + c0 & 0xffff) * 3 + pr >> 2;
    pr = squash->operator()(pr);
    return pr;
}

//////////////////////////// Encoder ////////////////////////////
//
// Encoder::code() calls predictor->predict_next_bit()/update() directly
// (a plain member pointer now, instead of a chunk-indexed global array).

class Encoder
{
private:
    const int mode;
    char *inout;
    size_t total_size;

    U32 x1, x2;
    U32 x;
    enum
    {
        BUFSIZE = 0x20000
    };
    U8 *buffer;
    size_t usize, csize;
    double usum, csum;
    Predictor *predictor;

public:
    size_t iterator_size;
    Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr, Predictor *predictor);
    ~Encoder();
    bool flush();
    bool put4(U32 c);

    int code(int y = 0)
    {
        assert(predictor);
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

    bool count()
    {
        assert(mode == COMPRESS);
        ++usize;
        if (csize > BUFSIZE - 256)
            return flush();
        return true;
    }
};

Encoder::Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr, Predictor *predictor) : mode(m), inout(temp), total_size(tsz), x1(0), x2(0xffffffff), x(0),
                                                                                                                 buffer(buffer_ptr), usize(0), csize(0), usum(0), csum(0), predictor(predictor), iterator_size(itr)
{
    if (mode == DECOMPRESS)
    { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (inout[iterator_size++] & 255);
        csize = 4;
    }
}
Encoder::~Encoder()
{
    buffer = 0;
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
        usum += (double)usize;
        csum += (double)csize + 10;
        x1 = x = 0;
        usize = csize = 0;
        x2 = 0xffffffff;
        return true;
    }
    return true;
}

size_t get4(size_t &itr, const char *in)
{
    size_t r = (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];

    return r;
}

//////////////////////// per-chunk resources ////////////////////////
//
// Mirrors ThreadBuffers + the second init<<<>>> kernel in the CUDA
// source, but scoped to a single chunk/thread instead of a global
// MAX_THREADS-sized array of device pointers. All the raw model-table
// buffers live in std::vectors owned here (freed automatically when a
// ChunkResources goes out of scope, same lifetime as one chunk on the
// GPU); the StateMap/Mix/APM/HashTable/LZP/Predictor wrapper objects
// are non-owning views over those buffers, exactly as in the original
// (their destructors null out the pointer, they never delete[] it).

struct ChunkResources
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

    std::unique_ptr<LZP> lzp_obj;
    std::unique_ptr<Predictor> predictor_obj;

    explicit ChunkResources(size_t MEM_host)
    {
        lzp_statemap.assign(0x200, 0);
        lzp_apm[0].assign(0x20000, 0);
        lzp_apm[1].assign(0x80000, 0);
        lzp_apm[2].assign(0x200000, 0);
        lzp_buffer.assign(MEM_host / 8, 0);
        lzp_table.assign(MEM_host / 32, 0);

        for (int j = 0; j < 11; j++)
            predictor_statemap[j].assign(0x100, 0);
        for (int j = 0; j < 10; j++)
            predictor_mix[j].assign(0x400, 0);
        predictor_apm[0].assign(0x20000, 0);
        predictor_apm[1].assign(0x20000, 0);
        predictor_apm[2].assign(0x20000, 0);
        predictor_hashtable.assign(MEM_host / 2 + 128, 0);
        predictor_context1.assign(0x40000, 0);

        StateMap *lzp_sm = new StateMap(lzp_statemap.data(), 0x200);
        APM *lzp_apm1 = new APM(lzp_apm[0].data(), 0x10000);
        APM *lzp_apm2 = new APM(lzp_apm[1].data(), 0x40000);
        APM *lzp_apm3 = new APM(lzp_apm[2].data(), 0x100000);
        lzp_obj.reset(new LZP(lzp_sm, lzp_buffer.data(), lzp_table.data(), lzp_apm1, lzp_apm2, lzp_apm3));

        StateMap *psm[11];
        for (int j = 0; j < 11; j++)
            psm[j] = new StateMap(predictor_statemap[j].data(), 0x100);
        Mix *pmix[10];
        for (int j = 0; j < 10; j++)
            pmix[j] = new Mix(predictor_mix[j].data(), 0x200);
        APM *papm1 = new APM(predictor_apm[0].data(), 0x10000);
        APM *papm2 = new APM(predictor_apm[1].data(), 0x10000);
        APM *papm3 = new APM(predictor_apm[2].data(), 0x10000);
        HashTable<16> *pht = new HashTable<16>((U32)(MEM_host / 2), predictor_hashtable.data());

        predictor_obj.reset(new Predictor(predictor_context1.data(), psm, pmix, papm1, papm2, papm3, pht, lzp_obj.get()));
    }
};

// One chunk's compress work -- direct translation of the COMPRESS branch
// of the CUDA paq9_cuda kernel body, minus warp broadcasting.
static void compressChunk(const char *input, size_t input_size, std::vector<char> &output, size_t &output_size, size_t MEM_host)
{
    ChunkResources res(MEM_host);
    Predictor *predictor = res.predictor_obj.get();
    LZP *lzp_obj = res.lzp_obj.get();
    std::vector<U8> encoder_buffer(0x20000);

    size_t itr = 0;
    Encoder encoder(COMPRESS, output.data(), encoder_buffer.data(), input_size, itr, predictor);
    bool store_mode = false;

    output[encoder.iterator_size++] = '0';

    itr = 0;
    while (itr < input_size)
    {
        int ch = (unsigned char)input[itr];
        itr++;

        int cp = lzp_obj->predict_char();
        if (ch == cp)
        {
            encoder.code(1);
        }
        else
        {
            for (int i = 8; i >= 0; --i)
                encoder.code(ch >> i & 1);
        }

        if (!encoder.count())
        {
            store_mode = true;
            break;
        }

        lzp_obj->update(ch);
    }

    bool flush_ok = encoder.flush();
    if (!flush_ok)
        store_mode = true;

    if (store_mode)
    {
        encoder.iterator_size = 0;
        output[encoder.iterator_size++] = '1';
        itr = 0;
        while (itr < input_size)
            output[encoder.iterator_size++] = input[itr++];
    }
    output_size = encoder.iterator_size;
}

// One chunk's decompress work -- direct translation of the DECOMPRESS
// branch of the CUDA paq9_cuda kernel body, minus warp broadcasting.
static void decompressChunk(const char *input, size_t input_size, std::vector<char> &output, size_t &output_size, size_t MEM_host)
{
    if (input_size > 0 && input[0] == '1')
    {
        size_t itr2 = 0;
        for (size_t i = 1; i < input_size; i++)
            output[itr2++] = input[i];
        output_size = itr2;
        return;
    }

    ChunkResources res(MEM_host);
    Predictor *predictor = res.predictor_obj.get();
    LZP *lzp_obj = res.lzp_obj.get();
    std::vector<U8> encoder_buffer(0x20000);

    size_t itr2 = 0; // output cursor
    size_t itr = 1;  // input cursor (byte 0 was the '0'/'1' marker)
    while (itr < input_size)
    {
        size_t usize = get4(itr, input);
        get4(itr, input); // csize, discarded (as in the original)

        Encoder encoder(DECOMPRESS, const_cast<char *>(input), encoder_buffer.data(), input_size, itr, predictor);

        size_t remaining = usize;
        while (remaining > 0)
        {
            --remaining;
            int cp = lzp_obj->predict_char();
            int first = encoder.code();
            if (first == 0)
            {
                cp = 1;
                while (cp < 256)
                    cp += cp + encoder.code();
                cp &= 255;
            }
            output[itr2++] = (char)cp;
            lzp_obj->update(cp);
        }

        itr = encoder.iterator_size;
    }
    output_size = itr2;
}

//////////////////////// host I/O helpers ////////////////////////

void put4_stream(U32 c, std::ostream &out)
{
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}

unsigned int get4_stream(std::istream &in)
{
    unsigned int r = (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
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
    size_t r = (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    return r;
}

bool check_archive(std::istream &in)
{
    std::string magic = "PAQ9-CUDA";

    for (char c : magic)
    {
        if (in.get() != c)
            return false;
    }

    return in.get() == 1;
}

std::string get_file_name_from_stream(std::istream &in)
{
    std::string file_name;
    char c;

    while (in.get(c) && c != '\0')
    {
        file_name += c;
    }

    return file_name;
}

const char *get_file_name(const char *path)
{
    const char *slash_pos = strrchr(path, '/');
    if (slash_pos)
        return slash_pos + 1;

#ifdef _WIN32
    const char *backslash_pos = strrchr(path, '\\');
    if (backslash_pos)
        return backslash_pos + 1;
#endif

    return path;
}

// One-time process-wide setup, equivalent to the CUDA source's
// init<<<1,1>>>() kernel (which built the single Squash/Stretch/Ilog
// instances and state_map_dt table shared by every chunk). Must run
// before any worker thread starts; everything it touches is read-only
// afterwards, so sharing it across threads is then safe.
void initGlobals()
{
    squash = new Squash();
    stretch = new Stretch();
    ilog = new Ilog(log_table);
    for (int i = 0; i < 1024; i++)
        state_map_dt[i] = 16384 / (i + i + 3);
}

static unsigned int pickThreadCount()
{
    unsigned int n = std::thread::hardware_concurrency();
    return n == 0 ? 4u : n;
}

//////////////////////////// compress / decompress ////////////////////////////

void compress(const char *destination_file, const char *source_file)
{
    std::ifstream source(source_file, std::ios::binary);
    if (!source)
    {
        std::cerr << "Cannot open " << source_file << endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_B = source.tellg();
    source.clear();
    source.seekg(0, std::ios::beg);

    chunk_MB = (1 << (chunk_level - 1));
    size_t chunk_B = (size_t)chunk_MB * MB;
    size_t num_of_chunks = (total_B + chunk_B - 1) / chunk_B;

    unsigned int num_of_thread = std::min(pickThreadCount(), (unsigned int)std::max<size_t>(num_of_chunks, 1));

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
    put4_stream((U32)chunk_MB, dest);
    put4_stream((U32)memory_level, dest);
    put4_stream((U32)chunk_level, dest);
    put4_stream((U32)num_of_chunks, dest);

    std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << chunk_level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;
    std::cout << "CPU worker threads: " << num_of_thread << endl;

    size_t MEM_host = 1ULL << (base_memory_level + memory_level);
    MEM = MEM_host; // must be set before any per-chunk LZP object is constructed

    total_compressed_size = 0;
    total_uncompressed_size = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t chunk_index = 0;
    while (chunk_index < num_of_chunks)
    {
        size_t batch = std::min<size_t>(num_of_thread, num_of_chunks - chunk_index);

        std::vector<std::vector<char>> inputs(batch);
        std::vector<size_t> input_sizes(batch);
        std::vector<std::vector<char>> outputs(batch);
        std::vector<size_t> output_sizes(batch, 0);

        for (size_t i = 0; i < batch; i++)
        {
            size_t current_B = std::min(chunk_B, total_B - (chunk_index + i) * chunk_B);
            inputs[i].resize(current_B);
            if (current_B > 0)
                source.read(inputs[i].data(), current_B);
            input_sizes[i] = current_B;
            outputs[i].assign(chunk_B + 2, 0);
        }

        std::atomic<size_t> next{0};
        std::vector<std::thread> workers;
        for (unsigned int t = 0; t < num_of_thread; t++)
        {
            workers.emplace_back([&]()
                                  {
                size_t i;
                while ((i = next.fetch_add(1)) < batch)
                    compressChunk(inputs[i].data(), input_sizes[i], outputs[i], output_sizes[i], MEM_host); });
        }
        for (auto &th : workers)
            th.join();

        size_t total_input = 0, total_output = 0;
        for (size_t i = 0; i < batch; i++)
        {
            put4_stream((U32)input_sizes[i], dest);
            put4_stream((U32)output_sizes[i], dest);
            dest.write(outputs[i].data(), output_sizes[i]);
            total_input += input_sizes[i];
            total_output += output_sizes[i];
        }
        total_uncompressed_size += total_input;
        total_compressed_size += total_output;

        chunk_index += batch;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Total execution time: " << elapsed.count() * 1000.0 << " ms" << endl;

    source.close();
    dest.close();

    std::cout << "Uncompressed  ->  Compressed\n";
    std::cout << "Total: " << total_uncompressed_size << " Byte -> " << total_compressed_size << " Byte" << endl;
    if (total_compressed_size > 0)
        std::cout << "Compression Ratio: " << 1.0 * total_uncompressed_size / total_compressed_size << endl;
}

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
    source.seekg(0, std::ios::end);
    source.clear();
    source.seekg(0, std::ios::beg);

    if (!check_archive(source))
    {
        std::cout << "This is not a PAQ9-CUDA compressed file.\n";
        exit(1);
    }

    std::string filename = get_file_name_from_stream(source);

    std::string out_name = destination_file ? std::string(destination_file) : filename;

    char mode = source.get();
    if (mode == 's')
    {
        // legacy stored-whole-file marker, kept for archive-format parity
        // with the original; the compressor here never emits it.
    }
    else if (mode == 'c')
    {
        size_t usize_total = get8_stream(source);
        (void)usize_total;
        chunk_MB = get4_stream(source);
        memory_level = get4_stream(source);
        chunk_level = get4_stream(source);
        size_t num_of_chunks = get4_stream(source);

        unsigned int num_of_thread = std::min(pickThreadCount(), (unsigned int)std::max<size_t>(num_of_chunks, 1));

        std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
        std::cout << "Memory Level: " << memory_level << endl;
        std::cout << "Level: " << chunk_level << endl;
        std::cout << "Number of Chunks: " << num_of_chunks << endl;
        std::cout << "CPU worker threads: " << num_of_thread << endl;

        size_t MEM_host = 1ULL << (base_memory_level + memory_level);
        MEM = MEM_host; // must be set before any per-chunk LZP object is constructed
        size_t chunk_B = (size_t)chunk_MB * MB;

        std::ofstream dest(out_name, std::ios::binary);
        if (!dest)
        {
            std::cout << out_name << " does not created/opened.\n";
            exit(1);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        size_t chunk_index = 0;
        while (chunk_index < num_of_chunks)
        {
            size_t batch = std::min<size_t>(num_of_thread, num_of_chunks - chunk_index);

            std::vector<std::vector<char>> inputs(batch);
            std::vector<size_t> input_sizes(batch);
            std::vector<std::vector<char>> outputs(batch);
            std::vector<size_t> output_sizes(batch, 0);

            for (size_t i = 0; i < batch; i++)
            {
                get4_stream(source); // per-chunk uncompressed size -- format parity only, see decompressChunk
                size_t isz = get4_stream(source);
                inputs[i].resize(isz);
                if (isz > 0)
                    source.read(inputs[i].data(), isz);
                input_sizes[i] = isz;
                outputs[i].assign(chunk_B + 2, 0);
            }

            std::atomic<size_t> next{0};
            std::vector<std::thread> workers;
            for (unsigned int t = 0; t < num_of_thread; t++)
            {
                workers.emplace_back([&]()
                                      {
                    size_t i;
                    while ((i = next.fetch_add(1)) < batch)
                        decompressChunk(inputs[i].data(), input_sizes[i], outputs[i], output_sizes[i], MEM_host); });
            }
            for (auto &th : workers)
                th.join();

            size_t total_input = 0, total_output = 0;
            for (size_t i = 0; i < batch; i++)
            {
                dest.write(outputs[i].data(), output_sizes[i]);
                total_input += input_sizes[i];
                total_output += output_sizes[i];
            }
            total_compressed_size += total_input;
            total_uncompressed_size += total_output;

            chunk_index += batch;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Total Execution Time: " << elapsed.count() << " seconds" << endl;

        dest.close();
        std::cout << "Compressed  ->  Decompressed \n";
        std::cout << "Total: " << total_compressed_size << " Byte -> " << total_uncompressed_size << " Byte" << endl;
    }
    source.close();
}

void print_usage(const char *prog_name)
{
    const char *file_name = get_file_name(prog_name);

    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << file_name << " -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>\n";
    std::cout << "  Decompress: " << file_name << " -d <source_file> <destination_file>\n\n";

    std::cout << "  <memory_level> and <chunk_level> must be between 1 and 11.\n";
    std::cout << "  If not given, or out of bounds, both default to 1.\n\n";

    std::cout << "  memory_level: controls how much RAM is used per chunk (model tables).\n";
    std::cout << "    - Use a HIGHER value if you have more RAM available,\n";
    std::cout << "      or if chunk_level is set higher (higher chunk levels need more memory).\n";
    std::cout << "    - Use a LOWER value if you have limited RAM.\n\n";

    std::cout << "  chunk_level: controls compression ratio vs. speed.\n";
    std::cout << "    - Use a HIGHER value for a better compression ratio (slower).\n";
    std::cout << "    - Use a LOWER value for faster, smaller (less thorough) compression.\n\n";

    std::cout << "Examples:\n";
    std::cout << "  " << file_name << " -c -8 output.paq -8 input.txt\n";
    std::cout << "  " << file_name << " -d output.paq input.txt\n\n";
    std::cout << "Note: [] is optional.\n";
    std::cout << "Run again and provide proper arguments.\n";
}

int main(int argc, char **args)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "CPU multi-threaded version of PAQ9 (ported from PAQ9-CUDA warp-cooperative) started successfully.\n\n";
    if (argc < 3)
    {
        print_usage(args[0]);
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
            print_usage(args[0]);
            exit(1);
        }
    }
    else
    {
        print_usage(args[0]);
        exit(1);
    }
    std::cout << "Working mode: "
              << (mode == COMPRESS ? "Compressing" : "Decompressing")
              << endl;

    initGlobals();

    int ind = 2;
    if (mode == COMPRESS)
    {
        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;

            size_t len = strlen(args[ind]);
            for (size_t i = 1; i < len; i++)
            {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            memory_level = stoi(temp);
            if (memory_level < 1 || memory_level > 11)
            {
                memory_level = 1;
                std::cout << "Your provided memory level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc)
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc)
        {
            destination_file_name = args[ind];
            ind++;
        }
        else
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;

            size_t len = strlen(args[ind]);
            for (size_t i = 1; i < len; i++)
            {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else
                {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            chunk_level = stoi(temp);
            if (chunk_level < 1 || chunk_level > 11)
            {
                chunk_level = 1;
                std::cout << "Your provided level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc)
        {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc)
        {
            source_file_name = args[ind];
            ind++;
        }
        else
        {
            print_usage(args[0]);
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
            print_usage(args[0]);
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
    if (seconds > 0 && total_uncompressed_size > 0)
        std::cout << "Compression/Decompression Speed: " << total_uncompressed_size / seconds / 1024 << " KB/seconds \n";

    return 0;
}
