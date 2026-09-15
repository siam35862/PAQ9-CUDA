// =====================================================================
// PAQ9-CUDA — warp-cooperative variant
//
// CHANGES vs the original single-thread-per-chunk version:
//   1. Each chunk is now processed by one WARP (32 raw threads) instead
//      of one raw thread. lane = threadIdx.x & 31, chunk = global_tid >> 5.
//   2. Predictor::predict_next_bit() / Predictor::update(): the 7 heavy
//      HashTable::operator[] lookups and the 11 StateMap lookups/updates
//      are independent of each other, so they are spread across lanes
//      4-10 and 0-10 respectively and issued concurrently. The Mix/APM
//      chains stay strictly serial (lane 0 only) because each stage
//      depends on the previous stage's output.
//   3. Encoder::code() now must be called by the WHOLE warp (because it
//      calls predict_next_bit()/update() which need all lanes). The
//      arithmetic-coder range state (x1,x2,csize,x) is still owned by
//      lane 0 only and broadcast with __shfl_sync where needed.
//   4. The main bit-processing loops in paq9_cuda are restructured to be
//      warp-convergent: the loop condition and the byte being processed
//      are decided by lane 0 and broadcast, so every lane enters/exits
//      the loop and calls encoder.code() together.
//   5. Host launch geometry: raw threads launched = 32 * num_of_chunks
//      instead of 1 * num_of_chunks. predictor[]/lzp[]/buffers[] arrays
//      are still indexed by CHUNK id (unchanged), only the paq9_cuda
//      launch config changes.
//
// IMPORTANT CAVEATS (read before using):
//   - This trades single-chunk latency for a 32x increase in raw threads
//     per chunk. It only pays off when you have FEW chunks (e.g. 1, as
//     in the small-file case that prompted this). With many chunks,
//     plain 1-thread-per-chunk parallelism is far more efficient — this
//     mode should ideally be selected conditionally (not done here, to
//     keep the diff focused).
//   - Warp-level primitives (__shfl_sync/__syncwarp) require ALL 32
//     lanes of the warp to reach the corresponding call in lockstep.
//     divergent early-returns inside a chunk boundary are avoided by
//     branching on `chunk >= num_of_chunks` BEFORE any warp primitive is
//     used, so an entire warp exits together, never partially.
//   - This is a nontrivial correctness-sensitive rewrite of an adaptive
//     arithmetic coder. A single race or lane-mismatch will silently
//     corrupt output rather than crash. ALWAYS verify with a full
//     compress -> decompress -> byte-diff round trip after building,
//     starting with small inputs, before trusting it on real data.
//   - Requires compute capability >= 3.0 for __shfl_sync/__syncwarp
//     (compile with -arch=sm_70 or higher recommended).
// =====================================================================

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring> // For strlen
#include <assert.h>
#include <cstdint>

// typedef
//  8, 16, 32 bit unsigned types (adjust as appropriate)
typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

// define
#define MAX_THREADS 2048
#define MAX_THREADS_PER_BLOCK 256
#define COMPRESS 0
#define DECOMPRESS 1
#define endl std::endl
#define HEAP_SIZE 64 // MB
constexpr size_t MB = 1024 * 1024;
#define base_memory_level 19 // default base memory level, MEM=1<<base_memory_level+memory_level
#define GPU_VRAM_LEVEL 4     // perchantage of VRAM , default 5 means 50% of VRAM will be used for compression
int memory_level = 1;        // default memory level MEM=1<<base_memory_level+memory_level;
int chunk_MB = 1;            // default memory chunks 1MB
int chunk_level = 1;
size_t total_uncompressed_size = 0;
size_t total_compressed_size = 0;

unsigned long long total_cuda_malloc_allocated = 0;

unsigned char *encoder_buffer[MAX_THREADS];

template <typename T>
cudaError_t cudaMallocTracked(T **pointer, size_t size)
{
    cudaError_t result = cudaMalloc(pointer, size);
    if (result == cudaSuccess)
    {
        total_cuda_malloc_allocated += size;
    }
    else
    {
        std::cout << "CudaMallocError to allocated with size : " << size << endl;
    }
    return result;
}

__device__ int get_tid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
class Squash
{
    short tab[4096];

public:
    __device__ Squash();
    __device__ int operator()(int d);
};

// intialize the sonstructor and method of Squash
__device__ Squash::Squash()
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
__device__ int Squash::operator()(int d)
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
__device__ Squash *squash;

//////////////////////////// Stretch ///////////////////////////////

class Stretch
{
    short t[4096];

public:
    __device__ Stretch();
    __device__ int operator()(int p) const;
};

__device__ Stretch::Stretch()
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

__device__ int Stretch::operator()(int p) const
{
    assert(p >= 0 && p < 4096);
    return t[p];
}

__device__ Stretch *stretch;

///////////////////////////// ilog //////////////////////////////

class Ilog
{
    U8 *table;

public:
    __device__ Ilog(U8 *table);
    __device__ int operator()(U16 x) const;
    __device__ int operator()(U32 x) const;
};

__device__ Ilog::Ilog(U8 *table) : table(table)
{
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i)
    {
        x += 774541002 / (i * 2 - 1); // numerator is 2^29/ln 2
        table[i] = x >> 24;
    }
}
__device__ int Ilog::operator()(U16 x) const
{
    return table[x];
}
__device__ int Ilog::operator()(U32 x) const
{
    if (x >= 0x1000000)
        return 256 + table[x >> 16];
    else if (x >= 0x10000)
        return 128 + table[x >> 8];
    else
        return table[x];
}
__device__ Ilog *ilog;

///////////////////////// state table ////////////////////////

__device__ static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, {15, 19}, {16, 23}, {17, 24}, {18, 25}, {20, 27}, {21, 28}, {22, 29}, {26, 30}, {31, 33}, {32, 35}, {32, 35}, {32, 35}, {32, 35}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {34, 37}, {36, 39}, {36, 39}, {36, 39}, {36, 39}, {38, 40}, {41, 43}, {42, 45}, {42, 45}, {44, 47}, {44, 47}, {46, 49}, {46, 49}, {48, 51}, {48, 51}, {50, 52}, {53, 43}, {54, 57}, {54, 57}, {56, 59}, {56, 59}, {58, 61}, {58, 61}, {60, 63}, {60, 63}, {62, 65}, {62, 65}, {50, 66}, {67, 55}, {68, 57}, {68, 57}, {70, 73}, {70, 73}, {72, 75}, {72, 75}, {74, 77}, {74, 77}, {76, 79}, {76, 79}, {62, 81}, {62, 81}, {64, 82}, {83, 69}, {84, 71}, {84, 71}, {86, 73}, {86, 73}, {44, 59}, {44, 59}, {58, 61}, {58, 61}, {60, 49}, {60, 49}, {76, 89}, {76, 89}, {78, 91}, {78, 91}, {80, 92}, {93, 69}, {94, 87}, {94, 87}, {96, 45}, {96, 45}, {48, 99}, {48, 99}, {88, 101}, {88, 101}, {80, 102}, {103, 69}, {104, 87}, {104, 87}, {106, 57}, {106, 57}, {62, 109}, {62, 109}, {88, 111}, {88, 111}, {80, 112}, {113, 85}, {114, 87}, {114, 87}, {116, 57}, {116, 57}, {62, 119}, {62, 119}, {88, 121}, {88, 121}, {90, 122}, {123, 85}, {124, 97}, {124, 97}, {126, 57}, {126, 57}, {62, 129}, {62, 129}, {98, 131}, {98, 131}, {90, 132}, {133, 85}, {134, 97}, {134, 97}, {136, 57}, {136, 57}, {62, 139}, {62, 139}, {98, 141}, {98, 141}, {90, 142}, {143, 95}, {144, 97}, {144, 97}, {68, 57}, {68, 57}, {62, 81}, {62, 81}, {98, 147}, {98, 147}, {100, 148}, {149, 95}, {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95}, {154, 107}, {108, 155}, {100, 156}, {157, 95}, {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105}, {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117}, {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183}, {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192}, {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115}, {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127}, {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219}, {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228}, {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125}, {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137}, {138, 247}, {140, 248}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {249, 135}, {250, 69}, {80, 251}, {140, 252}, {0, 0}, {0, 0}, {0, 0}};

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

__device__ int state_map_dt[1024];

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32 *prediction_table;

public:
    __device__ StateMap(U32 *prediction_table_ptr, int n = 256);
    __device__ ~StateMap();
    __device__ void update(int y, int limit = 255);
    __device__ int predict_next_bit(int cntx);
};

__device__ StateMap::StateMap(U32 *prediction_table_ptr, int n) : prediction_table(prediction_table_ptr), N(n), cntxt(0)
{
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
    if (state_map_dt[0] == 0)
        for (int i = 0; i < 1024; i++)
            state_map_dt[i] = 16384 / (i + i + 3);
}

__device__ StateMap::~StateMap()
{
    prediction_table = 0;
}

__device__ void StateMap::update(int y, int limit)
{
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10;

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] = prediction_table[cntxt] & 0xfffffc00 | limit;

    prediction_table[cntxt] += (((y << 22) - p) >> 3) * state_map_dt[n] & 0xfffffc00;
}

__device__ int StateMap::predict_next_bit(int cntx)
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
    __device__ Mix(int *weight_ptr, int n = 512);
    __device__ ~Mix();
    __device__ int prediction(int p1, int p2, int cntxt);
    __device__ void update(int y);
};

__device__ Mix::Mix(int *weight_ptr, int n) : wt(weight_ptr), N(n), x1(0), x2(0), context(0), last_prediction(0)
{
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

__device__ Mix::~Mix()
{
    wt = 0;
}

__device__ int Mix::prediction(int p1, int p2, int cntxt)
{
    assert(cntxt >= 0 & cntxt < N);
    context = cntxt * 2;
    return last_prediction = ((x1 = p1) * (wt[context] >> 16) + (x2 = p2) * (wt[context + 1] >> 16) + 128) >> 8;
}

__device__ void Mix::update(int y)
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
    __device__ APM(int *weight_ptr, int n);
};

__device__ APM::APM(int *weight_ptr, int n) : Mix(weight_ptr, n)
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
    __device__ HashTable(int n, U8 *table_ptr);
    __device__ ~HashTable();
    __device__ U8 *operator[](U32 i);
};

template <int B>
__device__ HashTable<B>::HashTable(int n, U8 *table_ptr) : table(table_ptr), raw_table(0), N(n)
{
    assert(B >= 2 && (B & B - 1) == 0);
    assert(N >= B * 4 && (N & N - 1) == 0);
    raw_table = table;
    table += 64 - int(reinterpret_cast<uintptr_t>(table) & 63);
}

template <int B>
__device__ U8 *HashTable<B>::operator[](U32 i)
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
__device__ HashTable<B>::~HashTable()
{
    raw_table = table = 0;
}

////////////////////////// LZP /////////////////////////

__device__ size_t MEM = 1 << (base_memory_level + 1);
__device__ inline bool isalpha_device(char ch)
{
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z');
}

__device__ inline char tolower_device(char ch)
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
    __device__ LZP(StateMap *statemap1, U8 *buffer, U32 *table, APM *apm1, APM *apm2, APM *apm3);
    __device__ ~LZP();
    __device__ int predict_char();
    __device__ int context(int i);
    __device__ int context4()
    {
        return hash2;
    }
    __device__ int context8()
    {
        return hash1;
    }
    __device__ int probability();
    __device__ void update(int ch);
};

__device__ LZP::LZP(StateMap *statemap, U8 *buf, U32 *tab, APM *apm1, APM *apm2, APM *apm3) : N(MEM / 8), H(MEM / 32),
                                                                                              match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0),
                                                                                              statemap(statemap), apm1(apm1), apm2(apm2), apm3(apm3),
                                                                                              literals(0), matches(0), word0(0), word1(0)
{
    assert(MEM > 0);
    assert(H > 0);
    buffer = buf;
    table = tab;
}

__device__ LZP::~LZP()
{
    delete statemap;
    delete apm1;
    delete apm2;
    delete apm3;
    table = 0;
    buffer = 0;
}

__device__ int LZP::predict_char()
{
    return len >= MINLEN ? buffer[match & N - 1] : -1;
}

__device__ int LZP::context(int i)
{
    assert(i > 0);
    return buffer[pos - i & N - 1];
}

__device__ int LZP::probability()
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

__device__ void LZP::update(int ch)
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
    if (isalpha_device(ch))
        word0 = word0 * (29 << 2) + tolower_device(ch);
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

__device__ LZP *lzp[MAX_THREADS];

//////////////////////////// Predictor /////////////////////////
//
// NOTE: predictor[]/lzp[] arrays are indexed by CHUNK id (not raw thread
// id). This was always true; it just now matters more since raw thread
// id and chunk id diverge (32 raw threads per chunk).

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
    int stretched_cache[N]; // warp-shared scratch: each lane's stretch()
                            // result, written by that lane, read by lane 0
                            // in the serial mix chain. Avoids illegal
                            // single-lane __shfl_sync calls.

public:
    __device__ Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr);
    __device__ ~Predictor();
    // These two are now WARP-COOPERATIVE: every lane of the calling warp
    // must invoke them together (they use __shfl_sync/__syncwarp inside).
    __device__ int predict_next_bit();
    __device__ void update(int y);
};

__device__ Predictor::Predictor(U8 *context1_ptr, StateMap *statemap1[N], Mix *mix1[N - 1], APM *apm1, APM *apm2, APM *apm3, HashTable<16> *hashtable_ptr) : c0(0), context1(context1_ptr), nibble(1), bcount(0),
                                                                                                                                                             apm1(apm1), apm2(apm2), apm3(apm3), hashtable(hashtable_ptr)
{
    for (int i = 0; i < N; ++i)
    {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1)
            mix[i] = mix1[i];
    }
}

__device__ Predictor::~Predictor()
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

// Update model — WARP-COOPERATIVE.
// Every lane 0..10 (lane < N) does its own statemap[lane]->update() and
// (for lane>=1) mix[lane-1]->update(); these are independent of each
// other. Only lane 0 owns c0/bcount/nibble and the APM chain (serial).
__device__ void Predictor::update(int y)
{
    assert(y == 0 || y == 1);
    int lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;

    if (c0 == 0)
    {
        if (lane == 0)
            c0 = 1 - y;
        c0 = __shfl_sync(mask, c0, 0);
        return;
    }

    if (lane == 0)
    {
        *sp[0] = nex(*sp[0], y);
        statemap[0]->update(y);
    }
    else if (lane < N)
    {
        *sp[lane] = nex(*sp[lane], y);
        statemap[lane]->update(y);
        mix[lane - 1]->update(y);
    }
    __syncwarp(mask);

    if (lane == 0)
    {
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
    // c0/bcount/nibble are only ever read by lane 0 in predict_next_bit(),
    // so no broadcast is needed here.
}

// Predict next bit — WARP-COOPERATIVE.
// Lanes 4-10 each issue one of the 7 heavy HashTable::operator[] lookups
// concurrently; lanes 0-10 each compute one StateMap::predict_next_bit()
// concurrently. The Mix/APM chains remain serial on lane 0 because each
// stage's output feeds the next.
__device__ int Predictor::predict_next_bit()
{
    int tid = get_tid();
    int lane = threadIdx.x & 31;
    int chunk = tid >> 5;
    const unsigned mask = 0xFFFFFFFFu;

    assert(lzp);
    if (c0 == 0)
    {
        // Single scalar result — no benefit from splitting across lanes,
        // but every lane still calls it together so lzp[]'s internal
        // state (if any were lane-sensitive) stays consistent. LZP itself
        // is only ever touched by lane 0 elsewhere by convention.
        int r = (lane == 0) ? lzp[chunk]->probability() : 0;
        return __shfl_sync(mask, r, 0);
    }

    // ---- lane 0 computes shared scalars, broadcasts to whole warp ----
    int pc = 0, r = 0, bc = 0;
    U32 c4 = 0, c8 = 0;
    if (lane == 0)
    {
        pc = lzp[chunk]->predict_char();
        r = pc + 256 >> 8 - bcount == c0;
        c4 = lzp[chunk]->context4();
        c8 = (lzp[chunk]->context8() << 4) - 1;
        bc = bcount;
    }
    pc = __shfl_sync(mask, pc, 0);
    r = __shfl_sync(mask, r, 0);
    c4 = __shfl_sync(mask, c4, 0);
    c8 = __shfl_sync(mask, c8, 0);
    bc = __shfl_sync(mask, bc, 0);

    if ((bc & 3) == 0)
    { // nibble boundary? update context pointers
        int pcr = pc & -r;
        U32 c4p = c4 << 8;

        if (bc == 0)
        { // byte boundary? update order-1 context pointers (cheap, lanes 0-3)
            if (lane == 0)
                cp[0] = context1 + (c4 >> 16 & 0xff00);
            if (lane == 1)
                cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
            if (lane == 2)
                cp[2] = context1 + (c4 & 0xff00) + 0x20000;
            if (lane == 3)
                cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
        }

        // 7 heavy HashTable lookups — independent, issued concurrently
        // on lanes 4-10.
        if (lane == 4)
            cp[4] = hashtable->operator[]((c4p & 0xffff00) - c0);
        if (lane == 5)
            cp[5] = hashtable->operator[]((c4p & 0xffffff00) * 3 + c0);
        if (lane == 6)
            cp[6] = hashtable->operator[](c4 * 7 + c0);
        if (lane == 7)
            cp[7] = hashtable->operator[]((c8 * 5 & 0xfffffc) + c0);
        if (lane == 8)
            cp[8] = hashtable->operator[]((c8 * 11 & 0xffffff0) + c0 + pcr * 13);
        if (lane == 9)
            cp[9] = hashtable->operator[]((lzp[chunk]->word0 * 5 + c0 + pcr * 17));
        if (lane == 10)
            cp[10] = hashtable->operator[]((lzp[chunk]->word1 * 7 + lzp[chunk]->word0 * 11 + c0 + pcr * 37));

        __syncwarp(mask); // make cp[] writes visible to all lanes before use
    }

    // ---- 11 StateMap predict_next_bit() calls — independent, parallel ----
    // Each participating lane writes its result into the warp-shared
    // stretched_cache[] (a Predictor member, so all lanes see it) instead
    // of trying to __shfl_sync a value out of a single active lane, which
    // is illegal (shfl_sync requires every lane in `mask` to execute the
    // same shfl instruction together; a value living only in one lane's
    // local variable cannot be pulled by a call that only that one lane
    // issues).
    r <<= 8;
    if (lane == 0)
    {
        sp[0] = &cp[0][c0];
        stretched_cache[0] = stretch->operator()(statemap[0]->predict_next_bit(*sp[0]));
    }
    else if (lane < N)
    {
        sp[lane] = &cp[lane][lane < 4 ? c0 : nibble];
        int st = *sp[lane];
        stretched_cache[lane] = stretch->operator()(statemap[lane]->predict_next_bit(st));
    }
    __syncwarp(mask); // make stretched_cache[] writes visible to lane 0

    // ---- serial Mix + APM chain: lane 0 only (each stage depends on
    // the previous stage's output, so this part cannot be parallelized) ----
    int pr = 0;
    if (lane == 0)
    {
        pr = stretched_cache[0];
        for (int i = 1; i < N; ++i)
        {
            int st_i = *sp[i];                    // lane i already wrote this above
            int stretched_i = stretched_cache[i]; // plain read, no shfl needed
            pr = mix[i - 1]->prediction(pr, stretched_i, st_i + r) * 3 + pr >> 2;
        }
        pr = apm1->prediction(512, pr * 2, c0 + pc * 256 & 0xffff) * 3 + pr >> 2;
        pr = apm2->prediction(512, pr * 2, c4 << 8 & 0xff00 | c0) * 3 + pr >> 2;
        pr = apm3->prediction(512, pr * 2, c4 * 3 + c0 & 0xffff) * 3 + pr >> 2;
        pr = squash->operator()(pr);
    }
    pr = __shfl_sync(mask, pr, 0); // called by ALL lanes (not guarded) — every
                                   // lane needs the same return value so the
                                   // caller's loop stays warp-convergent
    return pr;
}

__device__ Predictor *predictor[MAX_THREADS];

//////////////////////////// Encoder ////////////////////////////
//
// Encoder::code() now MUST be called by every lane of the warp, because
// it calls predictor->predict_next_bit()/update() which are warp
// cooperative. The arithmetic-coder range state (x1, x2, csize, x,
// iterator_size) is still logically owned by lane 0 only; other lanes
// just tag along so the warp-cooperative predictor calls stay convergent.

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

public:
    size_t iterator_size;
    __device__ Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr);
    __device__ ~Encoder();
    __device__ bool flush(); // lane 0 only, call with whole warp converged
    __device__ bool put4(U32 c);

    // Must be called by the WHOLE warp. Returns the same value y on
    // every lane.
    __device__ int code(int y = 0)
    {
        int lane = threadIdx.x & 31;
        const unsigned mask = 0xFFFFFFFFu;
        int tid = get_tid();
        int chunk = tid >> 5;

        assert(predictor);
        int p = predictor[chunk]->predict_next_bit(); // warp-cooperative call
        assert(p >= 0 && p < 4096);
        p += p < 2048;

        if (lane == 0)
        {
            U32 xmid = x1 + (x2 - x1 >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
            assert(xmid >= x1 && xmid < x2);
            if (mode == DECOMPRESS)
                y = x <= xmid;
            y ? (x2 = xmid) : (x1 = xmid + 1);
        }
        y = __shfl_sync(mask, y, 0); // broadcast the resolved bit to all lanes

        predictor[chunk]->update(y); // warp-cooperative call

        if (lane == 0)
        {
            while (((x1 ^ x2) & 0xff000000) == 0)
            { // pass equal leading bytes of range
                if (mode == COMPRESS)
                    buffer[csize++] = x2 >> 24;
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
                if (mode == DECOMPRESS)
                    x = (x << 8) + (inout[iterator_size++] & 255);
            }
        }
        __syncwarp(mask); // keep the warp converged before returning
        return y;
    }

    // Count one byte. Lane 0 owns usize/csize; call with whole warp
    // converged and use the broadcast return value for loop control.
    __device__ bool count()
    {
        int lane = threadIdx.x & 31;
        const unsigned mask = 0xFFFFFFFFu;
        int r = 1;
        if (lane == 0)
        {
            assert(mode == COMPRESS);
            ++usize;
            if (csize > BUFSIZE - 256)
                r = flush() ? 1 : 0;
        }
        r = __shfl_sync(mask, r, 0);
        return r != 0;
    }
};

__device__ Encoder::Encoder(int m, char *temp, unsigned char *buffer_ptr, size_t tsz, size_t itr) : mode(m), inout(temp), buffer(buffer_ptr), total_size(tsz), iterator_size(itr), x1(0), x2(0xffffffff), x(0),
                                                                                                    usize(0), csize(0), usum(0), csum(0)
{
    if (mode == DECOMPRESS)
    { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (inout[iterator_size++] & 255);
        csize = 4;
    }
}
__device__ Encoder::~Encoder()
{
    buffer = 0;
}

__device__ bool Encoder::put4(U32 c)
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

__device__ bool Encoder::flush()
{
    if (mode == COMPRESS)
    {
        buffer[csize++] = x1 >> 24;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        if (!put4(usize))
            return false;
        if (!put4(csize))
            return false;
        for (int i = 0; i < csize; i++)
        {
            if (iterator_size > total_size)
                return false;
            inout[iterator_size++] = buffer[i];
        }
        usum += usize;
        csum += csize + 10;
        x1 = x = usize = csize = 0;
        x2 = 0xffffffff;
        return true;
    }
    return true;
}

__device__ size_t get4(size_t &itr, const char *in)
{
    size_t r = (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];

    return r;
}

// =====================================================================
// paq9_cuda — restructured to be warp-convergent.
//
// chunk = global_raw_tid >> 5   (one warp == one chunk)
// lane  = threadIdx.x & 31
//
// Every lane in the warp must reach encoder.code() together (it's
// warp-cooperative). The loop condition and the byte being processed
// are decided by lane 0 and broadcast via __shfl_sync so all 32 lanes
// stay in lockstep.
// =====================================================================
__global__ void
paq9_cuda(
    size_t *input_size,
    char **input,
    size_t *output_size,
    char **output,
    unsigned char **buffer,
    int num_of_chunks, int mode, int memory_level)
{
    int tid = get_tid();
    int lane = threadIdx.x & 31;
    int chunk = tid >> 5;
    const unsigned mask = 0xFFFFFFFFu;

    // Whole warps exit together (chunk is the same for all 32 lanes of
    // a warp), so this branch never causes partial-warp divergence
    // across a __syncwarp/__shfl_sync boundary below.
    if (chunk >= num_of_chunks)
        return;

    if (lane == 0)
        output_size[chunk] = 100;

    if (mode == COMPRESS)
    {
        size_t itr = 0;
        Encoder encoder(mode, output[chunk], buffer[chunk], input_size[chunk], itr);
        int store_mode = 0;

        if (lane == 0)
            output[chunk][encoder.iterator_size++] = '0';

        itr = 0;
        while (true)
        {
            int cont = (lane == 0) ? (itr < input_size[chunk] ? 1 : 0) : 0;
            cont = __shfl_sync(mask, cont, 0);
            if (!cont)
                break;

            int ch = 0;
            if (lane == 0)
            {
                ch = (unsigned char)input[chunk][itr];
                itr++;
            }
            ch = __shfl_sync(mask, ch, 0);

            int cp = lzp[chunk]->predict_char(); // cheap, read-only; fine for every lane to call
            if (ch == cp)
            {
                encoder.code(1); // whole warp calls together
            }
            else
            {
                for (int i = 8; i >= 0; --i)
                    encoder.code(ch >> i & 1); // whole warp calls together
            }

            int ok = 1;
            if (!encoder.count()) // whole warp calls together, broadcasts result
                ok = 0;
            if (!ok)
            {
                store_mode = 1;
                break;
            }

            if (lane == 0)
                lzp[chunk]->update(ch);
            __syncwarp(mask);
        }

        int flush_ok = 1;
        if (lane == 0)
            flush_ok = encoder.flush() ? 1 : 0;
        flush_ok = __shfl_sync(mask, flush_ok, 0);
        if (!flush_ok)
            store_mode = 1;

        if (store_mode)
        {
            if (lane == 0)
            {
                encoder.iterator_size = 0;
                output[chunk][encoder.iterator_size++] = '1';
                itr = 0;
                while (itr < input_size[chunk])
                {
                    output[chunk][encoder.iterator_size++] = input[chunk][itr++];
                }
            }
        }
        if (lane == 0)
            output_size[chunk] = encoder.iterator_size;
    }
    else
    {
        // decompress
        if (input[chunk][0] == '1')
        {
            if (lane == 0)
            {
                int itr = 1, itr2 = 1;
                while (itr2 < input_size[chunk])
                {
                    output[chunk][itr++] = input[chunk][itr2++];
                }
                output_size[chunk] = itr;
            }
        }
        else
        {
            size_t itr2_shared = 0; // lane-0-owned running output offset
            size_t itr = 1;
            while (true)
            {
                int cont = (lane == 0) ? (itr < input_size[chunk] ? 1 : 0) : 0;
                cont = __shfl_sync(mask, cont, 0);
                if (!cont)
                    break;

                size_t usize = 0;
                if (lane == 0)
                {
                    usize = get4(itr, input[chunk]);
                    get4(itr, input[chunk]); // csize, discarded (as in original)
                }
                usize = __shfl_sync(mask, (unsigned long long)usize, 0);

                Encoder encoder(mode, input[chunk], buffer[chunk], input_size[chunk], itr);

                size_t remaining = usize;
                while (remaining > 0)
                {
                    --remaining;
                    int cp = lzp[chunk]->predict_char();
                    int first = encoder.code(); // whole warp calls together
                    if (first == 0)
                    {
                        cp = 1;
                        while (cp < 256)
                            cp += cp + encoder.code(); // whole warp calls together
                        cp &= 255;
                    }
                    if (lane == 0)
                    {
                        output[chunk][itr2_shared++] = cp;
                    }
                    if (lane == 0)
                        lzp[chunk]->update(cp);
                    __syncwarp(mask);
                }

                if (lane == 0)
                {
                    itr = encoder.iterator_size;
                    output_size[chunk] = itr2_shared;
                }
                itr = __shfl_sync(mask, (unsigned long long)itr, 0);
                __syncwarp(mask);
            }
        }
    }

    // Free this chunk's dynamically-allocated objects now that it's done,
    // so the device heap is returned for reuse. Only ONE lane per warp
    // must perform the delete (deleting the same pointer 32 times is
    // undefined behavior).
    __syncwarp(mask);
    if (lane == 0)
    {
        delete predictor[chunk];
        delete lzp[chunk];
        predictor[chunk] = 0;
        lzp[chunk] = 0;
    }
}

void put4(U32 c, int &iterator_size, char *inout)
{
    inout[iterator_size++] = char(c >> 24);
    inout[iterator_size++] = char(c >> 16);
    inout[iterator_size++] = char(c >> 8);
    inout[iterator_size++] = char(c);
}

unsigned int get4_stream(std::istream &in)
{
    unsigned int r = in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    return r;
}

void put4_stream(U32 c, std::ostream &out)
{
    out.put((c >> 24) & 0xFF);
    out.put((c >> 16) & 0xFF);
    out.put((c >> 8) & 0xFF);
    out.put(c & 0xFF);
}

size_t get8_stream(std::istream &in)
{
    size_t r = in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
    r = r * 256 + in.get();
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

size_t getMaximumFreeMemory()
{
    size_t free_byte = 0;
    size_t total_byte = 0;

    cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get memory info: " << cudaGetErrorString(err) << '\n';
        exit(1);
    }
    return free_byte;
}

struct ThreadBuffers
{
    U32 *lzp_statemap;
    int *lzp_apm[3];
    U8 *lzp_buffer;
    U32 *lzp_table;

    U32 *predictor_statemap[11];
    int *predictor_mix[10];
    int *predictor_apm[3];
    U8 *predictor_hashtable;
    U8 *predictor_context1;
};
__global__ void init(int memory_level, U8 *log_table)
{
    int tid = get_tid();
    if (tid == 0)
    {
        MEM = 1 << (base_memory_level + memory_level);
        squash = new Squash();
        stretch = new Stretch();
        ilog = new Ilog(log_table);
    }
}
// NOTE: this init kernel is still launched with ONE raw thread PER CHUNK
// (thread_count == number of chunks), unrelated to paq9_cuda's warp
// geometry, so it is unchanged.
__global__ void init(int thread_count, ThreadBuffers *buffers, int memory_level)
{
    int tid = get_tid();
    if (tid < thread_count)
    {
        ThreadBuffers &buffer = buffers[tid];
        StateMap *lzp_statemap = new StateMap(buffer.lzp_statemap, 0x200);
        APM *lzp_apm1 = new APM(buffer.lzp_apm[0], 0x10000);
        APM *lzp_apm2 = new APM(buffer.lzp_apm[1], 0x40000);
        APM *lzp_apm3 = new APM(buffer.lzp_apm[2], 0x100000);
        lzp[tid] = new LZP(lzp_statemap, buffer.lzp_buffer, buffer.lzp_table, lzp_apm1, lzp_apm2, lzp_apm3);

        StateMap *predictor_statemap[11];
        for (int j = 0; j < 11; j++)
            predictor_statemap[j] = new StateMap(buffer.predictor_statemap[j], 0x100);
        Mix *predictor_mix[10];
        for (int j = 0; j < 10; j++)
            predictor_mix[j] = new Mix(buffer.predictor_mix[j], 0x400);
        APM *predictor_apm1 = new APM(buffer.predictor_apm[0], 0x10000);
        APM *predictor_apm2 = new APM(buffer.predictor_apm[1], 0x10000);
        APM *predictor_apm3 = new APM(buffer.predictor_apm[2], 0x10000);
        HashTable<16> *predictor_hashtable = new HashTable<16>(MEM / 2, buffer.predictor_hashtable);
        predictor[tid] = new Predictor(buffer.predictor_context1, predictor_statemap, predictor_mix, predictor_apm1, predictor_apm2, predictor_apm3, predictor_hashtable);
    }
}
ThreadBuffers *buffers;

U8 *log_table;

// Pure calculator — no cudaMalloc, no ThreadBuffers/thread_count
// dependency. Returns the total bytes memoryAllocationForThread() would
// cudaMalloc, given only memory_level: log_table (allocated once) +
// one ThreadBuffers instance + one encoder_buffer slot.
size_t calculateThreadBufferBytes(int memory_level)
{
    U32 MEM_host = 1U << (base_memory_level + memory_level);

    size_t total = 0;

    // ---- allocated once, not per-thread ----
    total += 65536 * sizeof(U8); // log_table

    // ---- LZP ----
    total += 512 * sizeof(U32);             // lzp_statemap
    total += 131072 * sizeof(int);          // lzp_apm[0]
    total += 0x80000 * sizeof(int);         // lzp_apm[1]
    total += 0x200000 * sizeof(int);        // lzp_apm[2]
    total += (MEM_host / 8) * sizeof(U8);   // lzp_buffer   (memory_level dependent)
    total += (MEM_host / 32) * sizeof(U32); // lzp_table    (memory_level dependent)

    // ---- Predictor ----
    total += 11 * (0x100 * sizeof(U32));        // predictor_statemap[11]
    total += 10 * (0x800 * sizeof(int));        // predictor_mix[10]
    total += 3 * (0x20000 * sizeof(int));       // predictor_apm[3]
    total += (MEM_host / 2 + 128) * sizeof(U8); // predictor_hashtable (memory_level dependent)
    total += 0x40000 * sizeof(U8);              // predictor_context1

    // ---- Encoder scratch ----
    total += 0x20000 * sizeof(unsigned char); // encoder_buffer[i]

    return total;
}
void memoryAllocationForThread(int thread_count)
{

    cudaMallocTracked(&log_table, 65536 * sizeof(U8));
    init<<<1, 1>>>(memory_level, log_table);
    cudaDeviceSynchronize();

    U32 MEM_host = 1U << (base_memory_level + memory_level);

    cudaMallocManaged(&buffers, thread_count * sizeof(ThreadBuffers));
    total_cuda_malloc_allocated += thread_count * sizeof(ThreadBuffers);

    for (int i = 0; i < thread_count; i++)
    {
        cudaMallocTracked(&buffers[i].lzp_statemap, 512 * sizeof(U32));

        cudaMallocTracked(&buffers[i].lzp_apm[0], 131072 * sizeof(int));
        cudaMallocTracked(&buffers[i].lzp_apm[1], 0x80000 * sizeof(int));
        cudaMallocTracked(&buffers[i].lzp_apm[2], 0x200000 * sizeof(int));

        cudaMallocTracked(&buffers[i].lzp_buffer, MEM_host / 8 * sizeof(U8));
        cudaMallocTracked(&buffers[i].lzp_table, MEM_host / 32 * sizeof(U32));

        for (int j = 0; j < 11; j++)
            cudaMallocTracked(&buffers[i].predictor_statemap[j], 0x100 * sizeof(U32));

        for (int j = 0; j < 10; j++)
            cudaMallocTracked(&buffers[i].predictor_mix[j], 0x800 * sizeof(int));

        cudaMallocTracked(&buffers[i].predictor_apm[0], 0x20000 * sizeof(int));
        cudaMallocTracked(&buffers[i].predictor_apm[1], 0x20000 * sizeof(int));
        cudaMallocTracked(&buffers[i].predictor_apm[2], 0x20000 * sizeof(int));

        cudaMallocTracked(&buffers[i].predictor_hashtable,
                          (MEM_host / 2 + 128) * sizeof(U8));
        cudaMallocTracked(&buffers[i].predictor_context1, 0x40000 * sizeof(U8));
        cudaMallocTracked(&encoder_buffer[i], 0x20000 * sizeof(unsigned char));
    }
}
void deviceIntialization(int thread_count)
{
    int threadsPerBlock = 256;
    int blocks = (thread_count + threadsPerBlock - 1) / threadsPerBlock;
    init<<<blocks, threadsPerBlock>>>(thread_count, buffers, memory_level);
    cudaDeviceSynchronize();
}

__global__ void freeDeviceObjects(int thread_count)
{
    int tid = get_tid();

    if (tid < thread_count)
    {
        delete lzp[tid];
        delete predictor[tid];
    }

    if (tid == 0)
    {
        delete squash;
        delete stretch;
        delete ilog;
    }
}

void memoryDeallocationForThread(int thread_count)
{
    int threadsPerBlock = 256;
    int blocks = (thread_count + threadsPerBlock - 1) / threadsPerBlock;
    freeDeviceObjects<<<blocks, threadsPerBlock>>>(thread_count);
    cudaDeviceSynchronize();

    for (int i = 0; i < thread_count; i++)
    {
        cudaFree(buffers[i].lzp_statemap);

        cudaFree(buffers[i].lzp_apm[0]);
        cudaFree(buffers[i].lzp_apm[1]);
        cudaFree(buffers[i].lzp_apm[2]);

        cudaFree(buffers[i].lzp_buffer);
        cudaFree(buffers[i].lzp_table);

        for (int j = 0; j < 11; j++)
            cudaFree(buffers[i].predictor_statemap[j]);

        for (int j = 0; j < 10; j++)
            cudaFree(buffers[i].predictor_mix[j]);

        cudaFree(buffers[i].predictor_apm[0]);
        cudaFree(buffers[i].predictor_apm[1]);
        cudaFree(buffers[i].predictor_apm[2]);

        cudaFree(buffers[i].predictor_hashtable);
        cudaFree(buffers[i].predictor_context1);

        cudaFree(encoder_buffer[i]);
    }

    cudaFree(buffers);
    cudaFree(log_table);

    buffers = nullptr;
}

// Helper: compute (blocks, threadsPerBlock) for launching paq9_cuda with
// 32 raw threads per chunk. threadsPerBlock is kept a multiple of 32 so
// warps never straddle a chunk boundary.
static void computeWarpLaunchGeometry(int num_chunks, int &blocks, int &threadsPerBlock)
{
    threadsPerBlock = MAX_THREADS_PER_BLOCK; // 256 => 8 warps/block, multiple of 32
    long long total_raw_threads = 32LL * (long long)num_chunks;
    blocks = (int)((total_raw_threads + threadsPerBlock - 1) / threadsPerBlock);
    if (blocks < 1)
        blocks = 1;
}

void compress(char *destination_file, char *source_file)
{

    size_t maximum_memory = getMaximumFreeMemory();
    maximum_memory = GPU_VRAM_LEVEL * maximum_memory / 10;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE * MB);
    cudaError_t err1;
    err1 = cudaGetLastError();
    if (err1 != cudaSuccess)
    {
        std::cerr << "Launch error heap: "
                  << cudaGetErrorString(err1) << '\n';
        exit(1);
    }

    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess)
    {
        std::cerr << "Kernel error heap: "
                  << cudaGetErrorString(err1) << '\n';
        exit(1);
    }

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
    size_t memory_per_thread = 2 * chunk_MB * MB + calculateThreadBufferBytes(memory_level) + 1 * MB;

    int maximum_thread_per_device_call = (maximum_memory + memory_per_thread - 1) / memory_per_thread;

    size_t chunk_B = chunk_MB * MB;

    int num_of_chunks =
        (total_B + chunk_B - 1) / chunk_B;

    int device_call_count = (num_of_chunks + maximum_thread_per_device_call - 1) / maximum_thread_per_device_call;

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
    put4_stream(chunk_MB, dest);
    put4_stream(memory_level, dest);
    put4_stream(chunk_level, dest);
    put4_stream(num_of_chunks, dest);

    std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << chunk_level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;
    std::cout << "Total threads: " << num_of_chunks << endl;
    std::cout << "Maximum Thread at a time: " << maximum_thread_per_device_call << endl;

    int num_of_thread = std::min(maximum_thread_per_device_call, num_of_chunks);
    // Memory allocation for thread buffers and device objects
    auto start_time1 = std::chrono::high_resolution_clock::now();
    memoryAllocationForThread(num_of_thread);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time1 = end_time1 - start_time1;
    std::cout << "Memory Allocation Time: " << elapsed_time1.count() << " seconds" << endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Launch Kernel error: "
                  << cudaGetErrorString(err) << '\n';
        exit(1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Initialization kernel error: "
                  << cudaGetErrorString(err) << '\n';
        exit(1);
    }

    total_compressed_size = 0;
    total_uncompressed_size = 0;

    char **d_input;
    char **d_output;
    unsigned char **d_encoder_buffer;
    cudaMallocTracked(&d_input, num_of_thread * sizeof(char *));
    cudaMallocTracked(&d_output, num_of_thread * sizeof(char *));
    cudaMallocTracked(&d_encoder_buffer, num_of_thread * sizeof(unsigned char *));

    size_t *d_input_size;
    size_t *d_output_size;
    cudaMallocTracked(&d_input_size,
                      num_of_thread * sizeof(size_t));
    cudaMallocTracked(&d_output_size, num_of_thread * sizeof(size_t));

    char **temp_d_input =
        new char *[num_of_thread];

    char **temp_d_output =
        new char *[num_of_thread];

    for (int i = 0; i < num_of_thread; i++)
    {
        cudaMallocTracked(
            &temp_d_input[i],
            chunk_B * sizeof(char));

        cudaMallocTracked(
            &temp_d_output[i],
            (chunk_B + 2) * sizeof(char));
    }
    size_t *output_size = (size_t *)malloc(num_of_thread * sizeof(size_t));
    char **output = new char *[num_of_thread];

    for (int i = 0; i < num_of_thread; i++)
    {
        output[i] = new char[chunk_B + 2];
    }
    char **src_file = new char *[num_of_thread];
    for (int i = 0; i < num_of_thread; i++)
        src_file[i] = nullptr;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int call_count = 0; call_count < device_call_count; call_count++)
    {
        int num_of_current_thread = std::min(maximum_thread_per_device_call, (num_of_chunks - call_count * maximum_thread_per_device_call));

        std::vector<size_t> input_size(num_of_current_thread);

        for (size_t i = 0; i < num_of_current_thread; i++)
        {
            size_t current_B =
                min(chunk_B, total_B - ((call_count * maximum_thread_per_device_call) + i) * chunk_B);

            if (src_file[i] != nullptr)
                delete[] src_file[i];
            src_file[i] = new char[current_B];

            source.read(src_file[i], current_B);
            input_size[i] = current_B;
        }

        cudaMemcpy(
            d_input_size,
            input_size.data(),
            num_of_current_thread * sizeof(size_t),
            cudaMemcpyHostToDevice);

        for (int i = 0; i < num_of_current_thread; i++)
        {
            cudaMemcpy(
                temp_d_input[i],
                src_file[i],
                input_size[i] * sizeof(char),
                cudaMemcpyHostToDevice);
        }

        cudaMemcpy(
            d_input,
            temp_d_input,
            num_of_current_thread * sizeof(char *),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_output,
            temp_d_output,
            num_of_current_thread * sizeof(char *),
            cudaMemcpyHostToDevice);

        cudaMemcpy(
            d_encoder_buffer,
            encoder_buffer,
            num_of_current_thread * sizeof(unsigned char *),
            cudaMemcpyHostToDevice);

        // ---- Device Initialization (per-chunk kernel, unchanged geometry) ----
        auto init_start_time = std::chrono::high_resolution_clock::now();
        deviceIntialization(num_of_current_thread);
        cudaDeviceSynchronize();
        auto init_end_time = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end_time - init_start_time);
        std::cout << "Device initialization time: " << init_duration.count() << " ms" << endl;
        cudaError_t err1;
        err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Launch error during initialization: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Kernel error during initialization: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        ////////////////////paq9_cuda call (warp-cooperative geometry)////////////////////////////
        auto kernel_start_time = std::chrono::high_resolution_clock::now();
        std::cout << "input size: " << input_size[0] << " Byte, Current chunks: " << num_of_current_thread << endl;

        int blocks, threads;
        computeWarpLaunchGeometry(num_of_current_thread, blocks, threads);
        std::cout << "Warp-cooperative launch: " << blocks << " blocks x " << threads
                  << " threads (" << (32 * num_of_current_thread) << " raw threads for "
                  << num_of_current_thread << " chunks)" << endl;

        paq9_cuda<<<blocks, threads>>>(
            d_input_size,
            d_input,
            d_output_size,
            d_output, d_encoder_buffer,
            num_of_current_thread, COMPRESS, memory_level);

        cudaDeviceSynchronize();
        auto kernel_end_time = std::chrono::high_resolution_clock::now();
        auto kernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end_time - kernel_start_time);

        std::cout << "Kernel execution time: " << kernel_duration.count() << " ms" << endl;
        err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Launch error paq9: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Kernel error paq9: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        cudaMemcpy(output_size, d_output_size, num_of_current_thread * sizeof(size_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_of_current_thread; i++)
        {
            cudaMemcpy(
                output[i],
                temp_d_output[i],
                output_size[i] * sizeof(char),
                cudaMemcpyDeviceToHost);
        }

        size_t total_input = 0, total_output = 0;
        for (size_t i = 0; i < num_of_current_thread; i++)
        {
            total_input += input_size[i];
            total_output += output_size[i];
        }

        for (size_t i = 0; i < num_of_current_thread; i++)
        {
            put4_stream((U32)(input_size[i]), dest);
            put4_stream((U32)(output_size[i]), dest);
            dest.write(output[i], output_size[i]);
        }

        total_compressed_size += total_output;
        total_uncompressed_size += total_input;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() << " ms" << endl;
    for (int i = 0; i < num_of_thread; i++)
    {
        if (src_file[i] != nullptr)
            delete[] src_file[i];
        delete[] output[i];
    }
    delete[] src_file;
    delete[] output;
    free(output_size);
    source.close();
    dest.close();

    for (int i = 0; i < num_of_thread; i++)
    {
        cudaFree(temp_d_input[i]);
        cudaFree(temp_d_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_encoder_buffer);

    cudaFree(d_input_size);
    cudaFree(d_output_size);

    delete[] temp_d_input;
    delete[] temp_d_output;
    memoryDeallocationForThread(num_of_thread);
    std::cout << "Uncompressed  ->  Compressed\n";
    std::cout << "Total: " << total_uncompressed_size << " Byte -> " << total_compressed_size << " Byte" << endl;

    std::cout << "Compression Ratio: " << 1.0 * total_uncompressed_size / total_compressed_size << endl;
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
int get_number(std::istream &in)
{
    int number = 0;
    char c;

    while (in.get(c) && c != '\0')
    {
        if (c >= '0' && c <= '9')
            number = number * 10 + (c - '0');
    }

    return number;
}
std::string get_file_name(std::istream &in)
{
    std::string file_name;
    char c;

    while (in.get(c) && c != '\0')
    {
        file_name += c;
    }

    return file_name;
}
char *get_input(std::istream &source, size_t size)
{
    char *input = new char[size];

    source.read(input, size);

    return input;
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
    size_t total_B = source.tellg();
    source.clear();
    source.seekg(0, std::ios::beg);

    if (!check_archive(source))
    {
        std::cout << "This is not a PAQ9-CUDA compressed file.\n";
        exit(1);
    }

    std::string filename = get_file_name(source);

    if (destination_file == 0)
    {
        destination_file = filename.c_str();
    }

    char mode = source.get();
    if (mode == 's')
    {
    }
    else if (mode == 'c')
    {
        size_t maximum_memory = getMaximumFreeMemory();
        maximum_memory = GPU_VRAM_LEVEL * maximum_memory / 10;
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, HEAP_SIZE * MB);
        cudaError_t err1;
        err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Launch error heap: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess)
        {
            std::cerr << "Kernel error heap: "
                      << cudaGetErrorString(err1) << '\n';
            exit(1);
        }

        size_t usize = get8_stream(source);
        chunk_MB = get4_stream(source);
        memory_level = get4_stream(source);
        chunk_level = get4_stream(source);
        int num_of_chunks = get4_stream(source);

        size_t memory_per_thread = 2 * chunk_MB * MB + calculateThreadBufferBytes(memory_level) + 1 * MB;

        int maximum_thread_per_device_call = (maximum_memory + memory_per_thread - 1) / memory_per_thread;
        int device_call_count = (num_of_chunks + maximum_thread_per_device_call - 1) / maximum_thread_per_device_call;
        std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
        std::cout << "Memory Level: " << memory_level << endl;
        std::cout << "Level: " << chunk_level << endl;
        std::cout << "Number of Chunks: " << num_of_chunks << endl;
        std::cout << "Total threads: " << num_of_chunks << endl;
        std::cout << "Maximum Thread at a time: " << maximum_thread_per_device_call << endl;

        int num_of_thread = std::min(maximum_thread_per_device_call, num_of_chunks);
        // Memory allocation for thread buffers and device objects
        auto start_time1 = std::chrono::high_resolution_clock::now();
        memoryAllocationForThread(num_of_thread);
        auto end_time1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time1 = end_time1 - start_time1;
        std::cout << "Memory Allocation Time: " << elapsed_time1.count() << " seconds" << endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "Launch error: "
                      << cudaGetErrorString(err) << '\n';
            exit(1);
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            std::cerr << "Initialization kernel error: "
                      << cudaGetErrorString(err) << '\n';
            exit(1);
        }

        int chunk_B = chunk_MB * MB;
        char **d_input;
        char **d_output;

        cudaMallocTracked(&d_input, num_of_thread * sizeof(char *));
        cudaMallocTracked(&d_output, num_of_thread * sizeof(char *));

        size_t *d_input_size;
        size_t *d_output_size;

        cudaMallocTracked(&d_input_size,
                          num_of_thread * sizeof(size_t));

        cudaMallocTracked(&d_output_size, num_of_thread * sizeof(size_t));

        char **temp_d_input =
            new char *[num_of_thread];

        char **temp_d_output =
            new char *[num_of_thread];

        for (int i = 0; i < num_of_thread; i++)
        {
            cudaMallocTracked(
                &temp_d_input[i],
                (chunk_B + 2) * sizeof(char));

            cudaMallocTracked(
                &temp_d_output[i],
                (chunk_B) * sizeof(char));
        }
        size_t *output_size = (size_t *)malloc(num_of_thread * sizeof(size_t));

        char **output = new char *[num_of_thread];

        for (int i = 0; i < num_of_thread; i++)
        {
            output[i] = new char[chunk_B];
        }
        std::vector<char *> input(num_of_thread, nullptr);
        std::ofstream dest(destination_file, std::ios::binary);
        if (!dest)
        {
            std::cout << std::string(destination_file) << " does not created/opened.\n";
            exit(1);
        }

        for (int call_count = 0; call_count < device_call_count; call_count++)
        {
            int num_of_current_thread = std::min(maximum_thread_per_device_call, (num_of_chunks - call_count * maximum_thread_per_device_call));

            std::vector<size_t> input_size(num_of_current_thread);
            std::vector<size_t> uncompressed_size(num_of_current_thread);
            for (int i = 0; i < num_of_current_thread; i++)
            {
                if (input[i] != nullptr)
                    delete[] input[i];
                uncompressed_size[i] = get4_stream(source);
                input_size[i] = get4_stream(source);
                input[i] = get_input(source, input_size[i]);
            }

            cudaMemcpy(
                d_input_size,
                input_size.data(),
                num_of_current_thread * sizeof(size_t),
                cudaMemcpyHostToDevice);

            for (int i = 0; i < num_of_current_thread; i++)
            {
                cudaMemcpy(
                    temp_d_input[i],
                    input[i],
                    input_size[i] * sizeof(char),
                    cudaMemcpyHostToDevice);
            }

            cudaMemcpy(
                d_input,
                temp_d_input,
                num_of_current_thread * sizeof(char *),
                cudaMemcpyHostToDevice);

            cudaMemcpy(
                d_output,
                temp_d_output,
                num_of_current_thread * sizeof(char *),
                cudaMemcpyHostToDevice);

            // ---- Device Initialization (per-chunk kernel, unchanged geometry) ----
            auto init_start_time = std::chrono::high_resolution_clock::now();
            deviceIntialization(num_of_current_thread);
            auto init_end_time = std::chrono::high_resolution_clock::now();
            auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end_time - init_start_time);
            std::cout << "Device initialization time: " << init_duration.count() << " ms" << endl;
            ////////////////////paq9_cuda call (warp-cooperative geometry)///////////////////////////

            int blocks, threads;
            computeWarpLaunchGeometry(num_of_current_thread, blocks, threads);
            auto kernel_start_time = std::chrono::high_resolution_clock::now();
            paq9_cuda<<<blocks, threads>>>(
                d_input_size,
                d_input,
                d_output_size,
                d_output, encoder_buffer,
                num_of_current_thread, DECOMPRESS, memory_level);

            cudaDeviceSynchronize();
            auto kernel_end_time = std::chrono::high_resolution_clock::now();
            auto kernel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kernel_end_time - kernel_start_time);
            std::cout << "Kernel execution time: " << kernel_duration.count() << " ms" << endl;

            cudaError_t err1 = cudaGetLastError();
            if (err1 != cudaSuccess)
            {
                std::cerr << "Launch error paq9: "
                          << cudaGetErrorString(err1) << '\n';
                exit(1);
            }

            err1 = cudaDeviceSynchronize();
            if (err1 != cudaSuccess)
            {
                std::cerr << "Kernel error paq9: "
                          << cudaGetErrorString(err1) << '\n';
                exit(1);
            }

            cudaMemcpy(output_size, d_output_size, num_of_current_thread * sizeof(size_t), cudaMemcpyDeviceToHost);

            for (int i = 0; i < num_of_current_thread; i++)
            {
                cudaMemcpy(
                    output[i],
                    temp_d_output[i],
                    output_size[i] * sizeof(char),
                    cudaMemcpyDeviceToHost);
            }

            size_t total_input = 0, total_output = 0;

            for (size_t i = 0; i < num_of_current_thread; i++)
            {
                dest.write(output[i], output_size[i]);
                total_input += input_size[i];
                total_output += output_size[i];
            }
            total_compressed_size += total_input;
            total_uncompressed_size += total_output;
        }

        for (size_t i = 0; i < input.size(); ++i)
        {
            if (input[i] != nullptr)
                delete[] input[i];
            if (output[i] != nullptr)
                delete[] output[i];
        }
        free(output_size);
        delete[] output;

        source.close();
        dest.close();
        std::cout << "Compressed  ->  Decompressed \n";
        std::cout << "Total: " << total_compressed_size << " Byte -> "
                  << total_uncompressed_size << " Byte" << endl;

        for (int i = 0; i < num_of_thread; i++)
        {
            cudaFree(temp_d_input[i]);
            cudaFree(temp_d_output[i]);
        }

        cudaFree(d_input);
        cudaFree(d_output);

        cudaFree(d_input_size);
        cudaFree(d_output_size);

        delete[] temp_d_input;
        delete[] temp_d_output;

        memoryDeallocationForThread(num_of_thread);
    }
    else
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }
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

void print_usage(const char *prog_name)
{
    const char *file_name = get_file_name(prog_name);

    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << file_name << " -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>\n";
    std::cout << "  Decompress: " << file_name << " -d <source_file> <destination_file>\n\n";

    std::cout << "  <memory_level> and <chunk_level> must be between 1 and 11.\n";
    std::cout << "  If not given, or out of bounds, both default to 1.\n\n";

    std::cout << "  memory_level: controls how much GPU memory (VRAM) is used.\n";
    std::cout << "    - Use a HIGHER value if you have more VRAM available,\n";
    std::cout << "      or if chunk_level is set higher (higher chunk levels need more memory).\n";
    std::cout << "    - Use a LOWER value if you have limited VRAM.\n\n";

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
    std::cout << "CUDA version of PAQ9 (warp-cooperative) started successfully.\n\n";
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
    int ind = 2;
    if (mode == COMPRESS)
    {
        if (ind < argc && args[ind][0] == '-')
        {
            std::string temp;

            size_t len = strlen(args[ind]);
            for (int i = 1; i < len; i++)
            {
                if (isdigit(args[ind][i]))
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
            for (int i = 1; i < len; i++)
            {
                if (isdigit(args[ind][i]))
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

    cudaDeviceSynchronize();

    std::cout << "Total GPU memory allocated: "
              << total_cuda_malloc_allocated << " bytes ("
              << static_cast<double>(total_cuda_malloc_allocated) / (1024 * 1024)
              << " MiB)" << endl;

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Time Taken: "
              << seconds << " seconds\n";
    std::cout << "Compression/Decompression Speed: " << total_uncompressed_size / seconds / 1024 << " KB/seconds \n";

    return 0;
}