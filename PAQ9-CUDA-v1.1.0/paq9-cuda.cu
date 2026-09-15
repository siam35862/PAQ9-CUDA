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
#define MAX_THREADS 1024
#define MAX_THREADS_PER_BLOCK 256
#define COMPRESS 0
#define DECOMPRESS 1
#define endl std::endl
constexpr size_t MB = 1024 * 1024;
int memory_level = 1;       // default memory level MEM=1<<22+memory_level;
int memory_chunk_level = 1; // default memory chunks 1MB
int level = 1;
size_t total_uncompressed_size = 0;
size_t total_compressed_size = 0;
size_t maximum_heap_limit = 8;     // default heap limit
size_t maximum_free_memory = 1024; // default consider 1GB memory has free

__device__ unsigned long long device_call_heap_allocated = 0;
unsigned long long current_cuda_malloc_allocated = 0;
unsigned long long maximum_device_call_memory = 0;
unsigned long long maximum_cuda_malloc_for_device_call = 0;
unsigned long long maximum_device_heap_for_device_call = 0;

void begin_device_call_memory()
{
    unsigned long long zero = 0;
    current_cuda_malloc_allocated = 0;
    cudaMemcpyToSymbol(device_call_heap_allocated, &zero, sizeof(zero));
}

void finish_device_call_memory()
{
    unsigned long long device_heap_allocated = 0;
    cudaMemcpyFromSymbol(&device_heap_allocated, device_call_heap_allocated,
                         sizeof(device_heap_allocated));
    unsigned long long device_call_memory =
        current_cuda_malloc_allocated + device_heap_allocated;
    if (device_call_memory > maximum_device_call_memory)
    {
        maximum_device_call_memory = device_call_memory;
        maximum_cuda_malloc_for_device_call = current_cuda_malloc_allocated;
        maximum_device_heap_for_device_call = device_heap_allocated;
    }
}

template <typename T>
cudaError_t cudaMallocTracked(T **pointer, size_t size)
{
    cudaError_t result = cudaMalloc(pointer, size);
    if (result == cudaSuccess)
        current_cuda_malloc_allocated += size;
    return result;
}

__device__ int get_tid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

class Alloc
{
    long long total_allocated_size;

public:
    __device__ Alloc()
    {
        total_allocated_size = 0;
    }
    template <class T>
    __device__ void alloc(T *&p, size_t allocate_size)
    {
        p = new T[allocate_size]();
        if (!p)
        {
            printf("Error: Out of memory (failed to allocate %zu elements)\n", allocate_size * sizeof(T));
            total_allocated_size = -1;
            return;
        }
        total_allocated_size += allocate_size * sizeof(T);
        atomicAdd(&device_call_heap_allocated,
                  static_cast<unsigned long long>(allocate_size) * sizeof(T));
    }
};

__device__ Alloc *allocator[MAX_THREADS];

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

// Inverse of squash. stretch(d) returns ln(p/(1-p)), d scaled by 8 bits,
// p by 12 bits.  d has range -2047 to 2047 representing -8 to 8.
// p has range 0 to 4095 representing 0 to 1.

class Stretch
{
    short t[4096];

public:
    __device__ Stretch();
    __device__ int operator()(int p) const;
};
// intialize the sonstructor and method of Stretch

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

// global instance of Stretch
__device__ Stretch *stretch;

///////////////////////////// ilog //////////////////////////////

// ilog(x) = round(log2(x) * 16), 0 <= x < 64K

class Ilog
{
    U8 *t;

public:
    __device__ Ilog();
    __device__ int operator()(U16 x) const;
    __device__ int operator()(U32 x) const;
};

// intialize the sonstructor and method of Ilog

__device__ Ilog::Ilog()
{
    allocator[get_tid()]->alloc(t, 65536);
    U32 x = 14155776;
    for (int i = 2; i < 65536; ++i)
    {
        x += 774541002 / (i * 2 - 1); // numerator is 2^29/ln 2
        t[i] = x >> 24;
    }
}
__device__ int Ilog::operator()(U16 x) const
{
    return t[x];
}
// global instance of Ilog

__device__ Ilog *ilog;
__device__ int Ilog::operator()(U32 x) const
{
    if (x >= 0x1000000)
        return 256 + ilog->operator()(x >> 16);
    else if (x >= 0x10000)
        return 128 + ilog->operator()(x >> 8);
    else
        return ilog->operator()(x);
}

///////////////////////// state table ////////////////////////

// State table:
//   nex(state, 0) = next state if bit y is 0, 0 <= state < 256
//   nex(state, 1) = next state if bit y is 1
//
// States represent a bit history within some context.
// State 0 is the starting state (no bits seen).
// States 1-30 represent all possible sequences of 1-4 bits.
// States 31-252 represent a pair of counts, (n0,n1), the number
//   of 0 and 1 bits respectively.  If n0+n1 < 16 then there are
//   two states for each pair, depending on if a 0 or 1 was the last
//   bit seen.
// If n0 and n1 are too large, then there is no state to represent this
// pair, so another state with about the same ratio of n0/n1 is substituted.
// Also, when a bit is observed and the count of the opposite bit is large,
// then part of this count is discarded to favor newer data over old.

__device__ static const U8 State_table[256][2] = {
    {1, 2}, {3, 5}, {4, 6}, {7, 10}, {8, 12}, {9, 13}, {11, 14}, // 0
    {15, 19},
    {16, 23},
    {17, 24},
    {18, 25},
    {20, 27},
    {21, 28},
    {22, 29}, // 7
    {26, 30},
    {31, 33},
    {32, 35},
    {32, 35},
    {32, 35},
    {32, 35},
    {34, 37}, // 14
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {34, 37},
    {36, 39},
    {36, 39}, // 21
    {36, 39},
    {36, 39},
    {38, 40},
    {41, 43},
    {42, 45},
    {42, 45},
    {44, 47}, // 28
    {44, 47},
    {46, 49},
    {46, 49},
    {48, 51},
    {48, 51},
    {50, 52},
    {53, 43}, // 35
    {54, 57},
    {54, 57},
    {56, 59},
    {56, 59},
    {58, 61},
    {58, 61},
    {60, 63}, // 42
    {60, 63},
    {62, 65},
    {62, 65},
    {50, 66},
    {67, 55},
    {68, 57},
    {68, 57}, // 49
    {70, 73},
    {70, 73},
    {72, 75},
    {72, 75},
    {74, 77},
    {74, 77},
    {76, 79}, // 56
    {76, 79},
    {62, 81},
    {62, 81},
    {64, 82},
    {83, 69},
    {84, 71},
    {84, 71}, // 63
    {86, 73},
    {86, 73},
    {44, 59},
    {44, 59},
    {58, 61},
    {58, 61},
    {60, 49}, // 70
    {60, 49},
    {76, 89},
    {76, 89},
    {78, 91},
    {78, 91},
    {80, 92},
    {93, 69}, // 77
    {94, 87},
    {94, 87},
    {96, 45},
    {96, 45},
    {48, 99},
    {48, 99},
    {88, 101}, // 84
    {88, 101},
    {80, 102},
    {103, 69},
    {104, 87},
    {104, 87},
    {106, 57},
    {106, 57}, // 91
    {62, 109},
    {62, 109},
    {88, 111},
    {88, 111},
    {80, 112},
    {113, 85},
    {114, 87}, // 98
    {114, 87},
    {116, 57},
    {116, 57},
    {62, 119},
    {62, 119},
    {88, 121},
    {88, 121}, // 105
    {90, 122},
    {123, 85},
    {124, 97},
    {124, 97},
    {126, 57},
    {126, 57},
    {62, 129}, // 112
    {62, 129},
    {98, 131},
    {98, 131},
    {90, 132},
    {133, 85},
    {134, 97},
    {134, 97}, // 119
    {136, 57},
    {136, 57},
    {62, 139},
    {62, 139},
    {98, 141},
    {98, 141},
    {90, 142}, // 126
    {143, 95},
    {144, 97},
    {144, 97},
    {68, 57},
    {68, 57},
    {62, 81},
    {62, 81}, // 133
    {98, 147},
    {98, 147},
    {100, 148},
    {149, 95},
    {150, 107},
    {150, 107},
    {108, 151}, // 140
    {108, 151},
    {100, 152},
    {153, 95},
    {154, 107},
    {108, 155},
    {100, 156},
    {157, 95}, // 147
    {158, 107},
    {108, 159},
    {100, 160},
    {161, 105},
    {162, 107},
    {108, 163},
    {110, 164}, // 154
    {165, 105},
    {166, 117},
    {118, 167},
    {110, 168},
    {169, 105},
    {170, 117},
    {118, 171}, // 161
    {110, 172},
    {173, 105},
    {174, 117},
    {118, 175},
    {110, 176},
    {177, 105},
    {178, 117}, // 168
    {118, 179},
    {110, 180},
    {181, 115},
    {182, 117},
    {118, 183},
    {120, 184},
    {185, 115}, // 175
    {186, 127},
    {128, 187},
    {120, 188},
    {189, 115},
    {190, 127},
    {128, 191},
    {120, 192}, // 182
    {193, 115},
    {194, 127},
    {128, 195},
    {120, 196},
    {197, 115},
    {198, 127},
    {128, 199}, // 189
    {120, 200},
    {201, 115},
    {202, 127},
    {128, 203},
    {120, 204},
    {205, 115},
    {206, 127}, // 196
    {128, 207},
    {120, 208},
    {209, 125},
    {210, 127},
    {128, 211},
    {130, 212},
    {213, 125}, // 203
    {214, 137},
    {138, 215},
    {130, 216},
    {217, 125},
    {218, 137},
    {138, 219},
    {130, 220}, // 210
    {221, 125},
    {222, 137},
    {138, 223},
    {130, 224},
    {225, 125},
    {226, 137},
    {138, 227}, // 217
    {130, 228},
    {229, 125},
    {230, 137},
    {138, 231},
    {130, 232},
    {233, 125},
    {234, 137}, // 224
    {138, 235},
    {130, 236},
    {237, 125},
    {238, 137},
    {138, 239},
    {130, 240},
    {241, 125}, // 231
    {242, 137},
    {138, 243},
    {130, 244},
    {245, 135},
    {246, 137},
    {138, 247},
    {140, 248}, // 238
    {249, 135},
    {250, 69},
    {80, 251},
    {140, 252},
    {249, 135},
    {250, 69},
    {80, 251}, // 245
    {140, 252},
    {0, 0},
    {0, 0},
    {0, 0}}; // 252

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

// A StateMap maps a context to a probability.  Methods:
//
// Statemap sm(n) creates a StateMap with n contexts using 4*n bytes memory.
// sm.p(cx, limit) converts state cx (0..n-1) to a probability (0..4095)
//     that the next updated bit y=1.
//     limit (1..1023, default 255) is the maximum count for computing a
//     prediction.  Larger values are better for stationary sources.
// sm.update(y) updates the model with actual bit y (0..1).

__device__ int state_map_dt[1024];

class StateMap
{
protected:
    const int N;           // Number of contexts
    int cntxt;             // Context of last prediction
    U32 *prediction_table; // cntxt -> prediction in high 22 bits, count in low 10 bits

public:
    __device__ StateMap(int n = 256);
    __device__ ~StateMap(); // frees prediction_table

    // update bit y (0..1)
    __device__ void update(int y, int limit = 255);

    // predict next bit in context cntx

    __device__ int predict_next_bit(int cntx);
};

// Initialization

__device__ StateMap::StateMap(int n) : N(n), cntxt(0)
{

    allocator[get_tid()]->alloc(prediction_table, N);
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
    if (state_map_dt[0] == 0)
        for (int i = 0; i < 1024; i++)
            state_map_dt[i] = 16384 / (i + i + 3);
}

__device__ StateMap::~StateMap()
{
    delete[] prediction_table;
    prediction_table = 0;
}

__device__ void StateMap::update(int y, int limit)
{
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10; // count, prediction

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

// Mix combines 2 predictions and a context to produce a new prediction.
// Methods:
// Mix m(n) -- creates allowing with n contexts.
// m.pp(p1, p2, cx) -- inputs 2 stretched predictions and a context cx
//   (0..n-1) and returns a stretched prediction.  Stretched predictions
//   are fixed point numbers with an 8 bit fraction, normally -2047..2047
//   representing -8..8, such that 1/(1+exp(-p) is the probability that
//   the next update will be 1.
// m.update(y) updates the model after a prediction with bit y (0..1).

class Mix
{
protected:
    const int N;         // size
    int *wt;             // weights, scaled 24 bits
    int x1, x2;          // inputs, scaled 8 bits(-2047 to 2047)
    int context;         // last context
    int last_prediction; // last output

public:
    __device__ Mix(int n = 512);
    __device__ ~Mix(); // frees wt (APM inherits this destructor)
    __device__ int prediction(int p1, int p2, int cntxt);
    __device__ void update(int y);
};
// initialization

__device__ Mix::Mix(int n) : N(n), x1(0), x2(0), context(0), last_prediction(0)
{
    allocator[get_tid()]->alloc(wt, n * 2);
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

__device__ Mix::~Mix()
{
    delete[] wt;
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

// An APM is a Mix optimized for a constant in place of p1, used to
// refine a stretched prediction given a context cx.
// Normally p1 is in the range (0..4095) and p2 is doubled.

class APM : public Mix
{
public:
    __device__ APM(int n);
};

__device__ APM::APM(int n) : Mix(n)
{
    for (int i = 0; i < n; i++)
    {
        wt[2 * i] = 0;
    }
}

//////////////////////////// HashTable /////////////////////////

// A HashTable maps a 32-bit index to an array of B bytes.
// The first byte is a checksum using the upper 8 bits of the
// index.  The second byte is a priority (0 = empty) for hash
// replacement.  The index need not be a hash.
// HashTable<B> h(n) - create using n bytes  n and B must be
//     powers of 2 with n >= B*4, and B >= 2.
// h[i] returns array [1..B-1] of bytes indexed by i, creating and
//     replacing another element if needed.  Element 0 is the
//     checksum and should not be modified.

template <int B>
class HashTable
{
    U8 *table;     // table: 1 element= B bytes: checksuj priority data
    U8 *raw_table; // true address returned by alloc(), before cache-line alignment
    const U32 N;   // size in bytes

public:
    __device__ HashTable(int n);
    __device__ ~HashTable();
    __device__ U8 *operator[](U32 i);
};

template <int B>
__device__ HashTable<B>::HashTable(int n) : table(0), raw_table(0), N(n)
{
    assert(B >= 2 && (B & B - 1) == 0);
    assert(N >= B * 4 && (N & N - 1) == 0);
    allocator[get_tid()]->alloc(table, N + B * 4 + 64);
    raw_table = table;                                          // remember true allocation address
    table += 64 - int(reinterpret_cast<uintptr_t>(table) & 63); // align on cache line boundary
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
    int c = 0, c0 = 0;
    for (U32 i = 0; i < N; ++i)
    {
        if (table[i])
        {
            ++c;
            if (i % B == 0)
                ++c0;
        }
    }

    // printf("Thread No: %d :-HashTable<%d> %1.4f%% full, %1.4f%% utilized of %d KiB\n", get_tid(),
    //        B, 100.0 * c0 * B / N, 100.0 * c / N, N >> 10);
    delete[] raw_table; // must delete the original pointer, not the aligned one
    raw_table = table = 0;
}

////////////////////////// LZP /////////////////////////

__device__ size_t MEM = 1 << (22 + 1); // Global memory limit, 1 << 22+(memory option)
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
// LZP predicts the next byte and maintains context.  Methods:
// c() returns the predicted byte for the next update, or -1 if none.
// p() returns the 12 bit probability (0..4095) that c() is next.
// update(ch) updates the model with actual byte ch (0..255).
// c(i) returns the i'th prior byte of context, i > 0.
// c4() returns the order 4 context, shifted into the LSB.
// c8() returns a hash of the order 8 context, shifted 4 bits into LSB.
// word0, word1 are hashes of the current and previous word (a-z).

class LZP
{
private:
    const size_t N, H; // buffer, table size
    enum
    {
        MINLEN = 12
    }; // minimum match length
    U8 *buf;               // Rotating buffer of size N
    U32 *table;            // Hash Table of pointers in high 24 bits, state in low 8 bits
    int match;             // start of match
    size_t len;            // length of match
    size_t pos;            // position of next char to write to buffer
    U32 hash;              // context hash
    U32 hash1;             // hash of last 8 bytes updates, shifting 4 bits to MSB
    U32 hash2;             // last 4 updates, shifting 8 bits to MSB
    StateMap statemap1;    // len+offset->p
    APM apm1, apm2, apm3;  // p, context->p
    int literals, matches; // statistics
public:
    U32 word0, word1; // Hashes of last 2 words (case insensitive a-z)
    __device__ LZP();
    __device__ ~LZP();
    __device__ int predict_char(); // predicted char
    __device__ int context(int i); // context
    __device__ int context4()      // order 4 context, context(1) in LSB
    {
        return hash2;
    }
    __device__ int context8() // hash order 8 context
    {
        return hash1;
    }
    __device__ int probability();   // probability that next char is predict_char()*4096
    __device__ void update(int ch); // update model with actual char ch
};
// Initilization

__device__ LZP::LZP() : N(MEM / 8), H(MEM / 32),
                        match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0),
                        statemap1(0x200), apm1(0x10000), apm2(0x40000), apm3(0x100000),
                        literals(0), matches(0), word0(0), word1(0)
{
    assert(MEM > 0);
    assert(H > 0);
    allocator[get_tid()]->alloc(table, H);
    allocator[get_tid()]->alloc(buf, N);
}

// Print statistics
__device__ LZP::~LZP()
{
    int c = 0;
    for (int i = 0; i < H; ++i)
        c += (table[i] != 0);
    // printf("Thread No: %d :- LZP hash table %1.4f%% full of %d KiB\t"
    //        "LZP buffer %1.4f%% full of %d KiB\n",
    //        get_tid(),
    //        100.0 * c / H, H >> 8, pos < N ? 100.0 * pos / N : 100.0, N >> 10);
    // printf("Thread No: %d :- LZP %d literals, %d matches (%1.4f%% matched)\n", get_tid(),
    //        literals, matches,
    //        literals + matches > 0 ? 100.0 * matches / (literals + matches) : 0.0);
    delete[] table;
    delete[] buf;
    table = 0;
    buf = 0;
    // statemap1 and apm1/apm2/apm3 are member objects, not pointers:
    // their own destructors run automatically and free their internals.
}

// Predicted next byte, or -1 for no prediction
__device__ int LZP::predict_char()
{
    return len >= MINLEN ? buf[match & N - 1] : -1;
}

// Return i'th byte of context (i > 0)
__device__ int LZP::context(int i)
{
    assert(i > 0);
    return buf[pos - i & N - 1];
}

// Return prediction that c() will be the next byte (0..4095)
__device__ int LZP::probability()
{
    if (len < MINLEN)
        return 0;
    int cxt = static_cast<int>(len);
    if (len > 28)
        cxt = 28 + (len >= 32) + (len >= 64) + (len >= 128);
    int pc = predict_char();
    int pr = statemap1.predict_next_bit(cxt);
    pr = stretch->operator()(pr);
    pr = apm1.prediction(2048, pr * 2, hash2 * 256 + pc & 0xffff) * 3 + pr >> 2;
    pr = apm2.prediction(2048, pr * 2, hash1 * (11 << 6) + pc & 0x3ffff) * 3 + pr >> 2;
    pr = apm3.prediction(2048, pr * 2, hash1 * (7 << 4) + pc & 0xfffff) * 3 + pr >> 2;
    pr = squash->operator()(pr);
    return pr;
}

// Update model with predicted byte ch (0..255)
__device__ void LZP::update(int ch)
{
    int y = predict_char() == ch;      // 1 if prediction of ch was right, else 0
    hash1 = hash1 * (3 << 4) + ch + 1; // update context hashes
    hash2 = hash2 << 8 | ch;
    hash = hash * (5 << 2) + ch + 1 & H - 1;
    if (len >= MINLEN)
    {
        statemap1.update(y);
        apm1.update(y);
        apm2.update(y);
        apm3.update(y);
    }
    if (isalpha_device(ch))
        word0 = word0 * (29 << 2) + tolower_device(ch);
    else if (word0)
        word1 = word0, word0 = 0;
    buf[pos & N - 1] = ch; // update buf
    ++pos;
    if (y)
    { // extend match
        ++len;
        ++match;
        ++matches;
    }
    else
    { // find new match, try order 6 context first
        ++literals;
        y = 0;
        len = 1;
        match = table[hash];
        if (!((match ^ pos) & N - 1))
            --match;
        while (len <= 128 && buf[match - len & N - 1] == buf[pos - len & N - 1])
            ++len;
        --len;
    }
    table[hash] = pos;
}

__device__ LZP *lzp[MAX_THREADS];

//////////////////////////// Predictor /////////////////////////

// A Predictor estimates the probability that the next bit of
// uncompressed data is 1.  Methods:
// Predictor() creates.
// p() returns P(1) as a 12 bit number (0-4095).
// update(y) trains the predictor with the actual bit (0 or 1).

class Predictor
{
    enum
    {
        N = 11
    }; // number of contexts
    int c0;                  // last 0-7 bits with leading 1, 0 before LZP flag
    int nibble;              // last 0-3 bits with leading 1 (1..15)
    int bcount;              // number of bits in c0 (0..7)
    HashTable<16> hashtable; // context -> state
    StateMap statemap[N];    // state -> prediction
    U8 *cp[N];               // i -> state array of bit histories for i'th context
    U8 *sp[N];               // i -> pointer to bit history for i'th context
    Mix mix[N - 1];          // combines 2 predictions given a context
    APM apm1, apm2, apm3;    // adjusts a prediction given a context
    U8 *context1;            // order 1 contexts -> state

public:
    __device__ Predictor();
    __device__ ~Predictor(); // frees context1; member destructors free hashtable/statemap/mix/apm
    __device__ int predict_next_bit();
    __device__ void update(int y);
};

// Initialize
__device__ Predictor::Predictor() : c0(0), nibble(1), bcount(0), hashtable(MEM / 2),
                                    apm1(0x10000), apm2(0x10000), apm3(0x10000)
{
    allocator[get_tid()]->alloc(context1, 0x40000);
    for (int i = 0; i < N; ++i)
        sp[i] = cp[i] = context1;
}

// hashtable, statemap[N], mix[N-1] and apm1/apm2/apm3 free themselves via
// their own destructors when this object is destroyed; only context1
// (allocated directly by Predictor) needs freeing here.
__device__ Predictor::~Predictor()
{
    delete[] context1;
    context1 = 0;
}

// Update model
__device__ void Predictor::update(int y)
{
    assert(y == 0 || y == 1);
    assert(bcount >= 0 && bcount < 8);
    assert(c0 >= 0 && c0 < 256);
    assert(nibble >= 1 && nibble <= 15);
    if (c0 == 0)
        c0 = 1 - y;
    else
    {
        *sp[0] = nex(*sp[0], y);
        statemap[0].update(y);
        for (int i = 1; i < N; ++i)
        {
            *sp[i] = nex(*sp[i], y);
            statemap[i].update(y);
            mix[i - 1].update(y);
        }
        c0 += c0 + y;
        bcount++;
        if (bcount == 8)
            bcount = c0 = 0;
        if ((nibble += nibble + y) >= 16)
            nibble = 1;
        apm1.update(y);
        apm2.update(y);
        apm3.update(y);
    }
}

// Predict next bit
__device__ int Predictor::predict_next_bit()
{
    int tid = get_tid();
    assert(lzp);
    if (c0 == 0)
        return lzp[tid]->probability();
    else
    {

        // Set context pointers
        int pc = lzp[tid]->predict_char();        // mispredicted byte
        int r = pc + 256 >> 8 - bcount == c0;     // c0 consistent with mispredicted byte?
        U32 c4 = lzp[tid]->context4();            // last 4 whole context bytes, shifted into LSB
        U32 c8 = (lzp[tid]->context8() << 4) - 1; // hash of last 7 bytes with 4 trailing 1 bits
        if ((bcount & 3) == 0)
        { // nibble boundary?  Update context pointers
            pc &= -r;
            U32 c4p = c4 << 8;
            if (bcount == 0)
            { // byte boundary?  Update order-1 context pointers
                cp[0] = context1 + (c4 >> 16 & 0xff00);
                cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
                cp[2] = context1 + (c4 & 0xff00) + 0x20000;
                cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
            }
            cp[4] = hashtable[(c4p & 0xffff00) - c0];
            cp[5] = hashtable[(c4p & 0xffffff00) * 3 + c0];
            cp[6] = hashtable[c4 * 7 + c0];
            cp[7] = hashtable[(c8 * 5 & 0xfffffc) + c0];
            cp[8] = hashtable[(c8 * 11 & 0xffffff0) + c0 + pc * 13];
            cp[9] = hashtable[lzp[tid]->word0 * 5 + c0 + pc * 17];
            cp[10] = hashtable[lzp[tid]->word1 * 7 + lzp[tid]->word0 * 11 + c0 + pc * 37];
        }

        // Mix predictions
        r <<= 8;
        sp[0] = &cp[0][c0];
        int pr = stretch->operator()(statemap[0].predict_next_bit(*sp[0]));
        for (int i = 1; i < N; ++i)
        {
            sp[i] = &cp[i][i < 4 ? c0 : nibble];
            int st = *sp[i];
            pr = mix[i - 1].prediction(pr, stretch->operator()(statemap[i].predict_next_bit(st)), st + r) * 3 + pr >> 2;
        }
        pr = apm1.prediction(512, pr * 2, c0 + pc * 256 & 0xffff) * 3 + pr >> 2; // Adjust prediction
        pr = apm2.prediction(512, pr * 2, c4 << 8 & 0xff00 | c0) * 3 + pr >> 2;
        pr = apm3.prediction(512, pr * 2, c4 * 3 + c0 & 0xffff) * 3 + pr >> 2;
        return squash->operator()(pr);
    }
}

__device__ Predictor *predictor[MAX_THREADS];

//////////////////////////// Encoder ////////////////////////////

// An Encoder arithmetic codes in blocks of size BUFSIZE.  Methods:
// Encoder(COMPRESS, f) creates encoder for compression to archive f, which
//     must be open past any header for writing in binary mode.
// Encoder(DECOMPRESS, f) creates encoder for decompression from archive f,
//     which must be open past any header for reading in binary mode.
// code(i) in COMPRESS mode compresses bit i (0 or 1) to file f.
// code() in DECOMPRESS mode returns the next decompressed bit from file f.
// count() should be called after each byte is compressed.
// flush() should be called after compression is done.  It is also called
//   automatically when a block is written.

class Encoder
{
private:
    const int mode; // Compress or decompress?
    char *inout;
    size_t total_size;

    U32 x1, x2; // Range, initially [0, 1), scaled by 2^32
    U32 x;      // Decompress mode: last 4 input bytes of archive
    enum
    {
        BUFSIZE = 0x20000
    };
    unsigned char *buf;  // Compression output buffer, size BUFSIZE
    size_t usize, csize; // Buffered uncompressed and compressed sizes
    double usum, csum;   // Total of usize, csize

public:
    size_t iterator_size;
    __device__ Encoder(int m, char *temp, size_t tsz, size_t itr);
    __device__ ~Encoder();   // frees buf (COMPRESS mode only; inout is not owned by Encoder)
    __device__ bool flush(); // call this when compression is finished
    __device__ bool put4(U32 c);

    // Compress bit y or return decompressed bit
    __device__ int code(int y = 0)
    {
        int tid = get_tid();
        assert(predictor);
        int p = predictor[tid]->predict_next_bit();
        assert(p >= 0 && p < 4096);
        p += p < 2048;
        U32 xmid = x1 + (x2 - x1 >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
        assert(xmid >= x1 && xmid < x2);
        if (mode == DECOMPRESS)
            y = x <= xmid;
        y ? (x2 = xmid) : (x1 = xmid + 1);
        predictor[tid]->update(y);
        while (((x1 ^ x2) & 0xff000000) == 0)
        { // pass equal leading bytes of range
            if (mode == COMPRESS)
                buf[csize++] = x2 >> 24;
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
            if (mode == DECOMPRESS)
                x = (x << 8) + (inout[iterator_size++] & 255);
            ;
        }
        return y;
    }

    // Count one byte
    __device__ bool count()
    {
        assert(mode == COMPRESS);
        ++usize;
        if (csize > BUFSIZE - 256)
            return flush();
        return true;
    }
};

// Create in mode m (COMPRESS or DECOMPRESS) with f opened as the archive.
__device__ Encoder::Encoder(int m, char *temp, size_t tsz, size_t itr) : mode(m), inout(temp), total_size(tsz), iterator_size(itr), x1(0), x2(0xffffffff), x(0),
                                                                         usize(0), csize(0), usum(0), csum(0)
{
    int tid = get_tid();
    buf = 0;
    if (mode == DECOMPRESS)
    { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (inout[iterator_size++] & 255);
        csize = 4;
    }
    else if (!buf)
        allocator[tid]->alloc(buf, BUFSIZE);
}

__device__ Encoder::~Encoder()
{
    if (mode == COMPRESS && buf)
    {
        delete[] buf;
        buf = 0;
    }
    // inout is owned by the caller (points into the chunk's device buffer) - never freed here.
}

// write 4 byte in inout
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

// Write a compressed block and reinitialize the encoder.  The format is:
//   uncompressed size (usize, 4 byte, MSB first)
//   compressed size (csize, 4 bytes, MSB first)
//   compressed data (csize bytes)
__device__ bool Encoder::flush()
{
    if (mode == COMPRESS)
    {
        buf[csize++] = x1 >> 24;
        buf[csize++] = 255;
        buf[csize++] = 255;
        buf[csize++] = 255;
        // inout[iterator_size++] = 0;   // putc(0, archive);
        // inout[iterator_size++] = 'c'; // putc('c', archive);
        if (!put4(usize))
            return false;
        if (!put4(csize))
            return false;
        for (int i = 0; i < csize; i++)
        {
            if (iterator_size > total_size)
                return false;
            inout[iterator_size++] = buf[i];
        }
        usum += usize;
        csum += csize + 10;
        // printf("%15.0f -> %15.0f"
        //        "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b",
        //    usum, csum);
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

__global__ void init(int memory_level)
{
    // blockDim.x == 0 বাদ দেওয়া হয়েছে
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        squash = new Squash();
        stretch = new Stretch();
        allocator[get_tid()] = new Alloc();
        ilog = new Ilog();
        MEM = 1 << (22 + memory_level);
        delete allocator[get_tid()];
    }
}

__global__ void
paq9_cuda(
    size_t *input_size,
    char **input,
    size_t *output_size,
    char **output,
    int num_of_chunks, int mode, int memory_level)
{
    int tid = get_tid();

    if (tid >= num_of_chunks)
        return;

    allocator[tid] = new Alloc();
    predictor[tid] = new Predictor();
    lzp[tid] = new LZP();

    // printf("%8d KiB\b\b\b\b\b\b\b\b\b\b\b\b", allocated >> 10);

    // // COmpress
    if (mode == COMPRESS)
    {

        int itr = 0;
        Encoder encoder(mode, output[tid], input_size[tid], itr);
        int ch;
        output[tid][encoder.iterator_size++] = '0';
        itr = 0;
        int store_mode = 0;
        while (itr < input_size[tid])
        {
            ch = (unsigned char)input[tid][itr];
            itr++;

            int cp = lzp[tid]->predict_char();
            if (ch == cp)
                encoder.code(1);
            else
                for (int i = 8; i >= 0; --i)
                    encoder.code(ch >> i & 1);
            if (!encoder.count())
            {
                store_mode = 1;
                break;
            }
            lzp[tid]->update(ch);
        }
        if (!encoder.flush())
        {
            store_mode = 1;
        }
        if (store_mode)
        {
            encoder.iterator_size = 0;
            output[tid][encoder.iterator_size++] = '1';

            itr = 0;
            while (itr < input_size[tid])
            {
                output[tid][encoder.iterator_size++] = input[tid][itr++];
            }
        }
        output_size[tid] = encoder.iterator_size;
    }
    else
    {
        int itr2 = 0;
        // decompress

        if (input[tid][0] == '1')
        {
            int itr = 0;
            itr2 = 1;
            while (itr2 < input_size[tid])
            {
                output[tid][itr++] = input[tid][itr2++];
            }
            output_size[tid] = itr;
        }
        else
        {
            size_t usize;
            itr2 = 0;
            size_t itr = 1;
            while (itr < input_size[tid])
            {
                usize = get4(itr, input[tid]);
                get4(itr, input[tid]); // csize
                Encoder encoder(mode, input[tid], input_size[tid], itr);

                while (usize--)
                {

                    int cp = lzp[tid]->predict_char();
                    if (encoder.code() == 0)
                    {
                        cp = 1;
                        while (cp < 256)
                            cp += cp + encoder.code();
                        cp &= 255;
                    }
                    output[tid][itr2++] = cp;
                    lzp[tid]->update(cp);
                }
                itr = encoder.iterator_size;

                output_size[tid] = itr2;
            }
        }
    }

    // Free this thread's dynamically-allocated objects now that its chunk
    // is done, so the device heap (cudaLimitMallocHeapSize) is returned for
    // reuse by other blocks instead of staying held for the whole kernel.
    // deleting predictor[tid] and lzp[tid] cascades: their member objects
    // (StateMap, Mix, APM, HashTable) each free their own internal arrays
    // via the destructors added above.
    delete predictor[tid];
    delete lzp[tid];
    delete allocator[tid];
    predictor[tid] = 0;
    lzp[tid] = 0;
    allocator[tid] = 0;
}
void put4(U32 c, int &iterator_size, char *inout)
{
    inout[iterator_size++] = char(c >> 24);
    inout[iterator_size++] = char(c >> 16);
    inout[iterator_size++] = char(c >> 8);
    inout[iterator_size++] = char(c);
}

// Read/write a 4 byte big-endian number from file
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

//// Read/write a 8 byte big-endian number from file
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

// Memory Level calculation

size_t getBytesFromMemoryLevel(int n)
{
    // n বৈধ কিনা চেক
    if (n < 1 || n > 9)
        return SIZE_MAX; // ভুল ইনপুট — unsigned, তাই -1 ব্যবহার করা যাবে না

    size_t MEM = (size_t)1 << (22 + n); // MEM = 2^(22+n)
    size_t total_bytes = (size_t)(0.75 * MEM) + 13030528ULL;

    return total_bytes;
}

int getMemoryLevelFromBytes(size_t bytes)
{
    for (int n = 1; n <= 9; ++n)
    {
        size_t threshold = getBytesFromMemoryLevel(n);
        if (bytes <= threshold)
            return n;
    }
    return 9; // bytes যদি সর্বোচ্চ থ্রেশহোল্ডও ছাড়িয়ে যায়, সর্বোচ্চ n রিটার্ন
}

size_t getMaximumHeapLimit()
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cerr << "No CUDA-capable NVIDIA GPU found.\n";
        exit(1);
    }

    int device = 0;
    cudaDeviceProp prop;

    err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get CUDA device properties: "
                  << cudaGetErrorString(err) << '\n';
        exit(1);
    }

    size_t low = 0;
    size_t high = prop.totalGlobalMem;
    size_t bestLimit = 0;

    // Binary search to find the maximum allowed heap size
    while (low <= high)
    {
        size_t mid = low + (high - low) / 2;

        cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, mid);

        if (err == cudaSuccess)
        {
            bestLimit = mid;
            low = mid + 1; // Try a larger size
        }
        else
        {
            // If high is at the max possible value to avoid underflow
            if (high == 0 || mid == 0)
                break;
            high = mid - 1; // Try a smaller size
        }
    }

    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, bestLimit);

    if (err != cudaSuccess)
    {
        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess)
            std::cerr << "Heap Launch error: "
                      << cudaGetErrorString(err1) << '\n';
        exit(1);
    }
    return bestLimit;
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

void compress(char *destination_file, char *source_file)
{
    // std::cout << "Compression cooking........." << endl;

    maximum_heap_limit = getMaximumHeapLimit();
    maximum_free_memory = getMaximumFreeMemory();

    size_t maximum_memory;

    maximum_memory = (size_t)4095 * 1024 * 1024; // 4GB
    maximum_memory = std::min(maximum_free_memory, maximum_heap_limit);
    maximum_memory = 5 * maximum_memory / 10;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, maximum_memory);
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

    memory_chunk_level = (1 << (level - 1));
    memory_level = getMemoryLevelFromBytes(memory_chunk_level * MB);
    size_t memory_per_thread = 3.5 * getBytesFromMemoryLevel(memory_level);

    int maximum_thread_per_device_call = (maximum_memory + memory_per_thread - 1) / memory_per_thread;

    size_t chunk_B = memory_chunk_level * MB;

    // std::cout << memory_chunk_level << " " << memory_level << " " << memory_per_thread << " " << maximum_thread_per_device_call << " " << chunk_B << endl;

    int num_of_chunks =
        (total_B + chunk_B - 1) / chunk_B;

    int device_call_count = (num_of_chunks + maximum_thread_per_device_call - 1) / maximum_thread_per_device_call;

    // compressed file configuration

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest)
    {
        std::cout << std::string(destination_file) << " does not created/opened.\n";
        exit(1);
    }

    std::string lvl = std::to_string(memory_level);
    dest.write("PAQ9-CUDA", 9);                   // program name
    dest.put(1);                                  // program version
    dest.write(source_file, strlen(source_file)); // filename
    dest.put(0);
    dest.put('c');              // compressed mode
    put8_stream(total_B, dest); // total uncompressed size in bytes
    put4_stream(memory_chunk_level, dest);
    put4_stream(memory_level, dest);
    put4_stream(level, dest);
    put4_stream(num_of_chunks, dest); // num of chunks
    // put4_stream(device_call_count, dest);          // total device call
    // put4_stream(maximum_thread_per_device_call, dest); // maximum thread per device call

    // std out
    std::cout << "Memory Chunk Level: " << memory_chunk_level << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;
    std::cout << "Total threads: " << num_of_chunks << endl;
    std::cout << "Maximum Thread at a time: " << maximum_thread_per_device_call << endl;
    // std::cout << "Maximum processed per device call: "
    //           << maximum_thread_per_device_call * chunk_B / MB << " MB" << endl;
    // std::cout << "Total Device Call " << device_call_count << endl;

    // device initialization
    init<<<1, 1>>>(memory_level);
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
    for (int call_count = 0; call_count < device_call_count; call_count++)
    {
        // std::cout << "\n\nDevice Call No: " << call_count + 1 << endl;
        int num_of_current_thread = std::min(maximum_thread_per_device_call, (num_of_chunks - call_count * maximum_thread_per_device_call));

        char **src_file = new char *[num_of_current_thread];

        std::vector<size_t> input_size(num_of_current_thread);

        for (size_t i = 0; i < num_of_current_thread; i++)
        {
            size_t current_B =
                min(chunk_B, total_B - ((call_count * maximum_thread_per_device_call) + i) * chunk_B);

            src_file[i] = new char[current_B];

            source.read(src_file[i], current_B);
            input_size[i] = current_B;
        }

        // preparing for calling device function

        // --------------------------------------------------
        // Device pointer arrays
        // --------------------------------------------------

        char **d_input;
        char **d_output;

        begin_device_call_memory();
        cudaMallocTracked(&d_input, num_of_current_thread * sizeof(char *));
        cudaMallocTracked(&d_output, num_of_current_thread * sizeof(char *));

        // --------------------------------------------------
        // Size arrays
        // --------------------------------------------------

        size_t *d_input_size;
        size_t *d_output_size;

        cudaMallocTracked(&d_input_size,
                          num_of_current_thread * sizeof(size_t));

        cudaMallocTracked(&d_output_size, num_of_current_thread * sizeof(size_t));
        // --------------------------------------------------
        // Copy input sizes: HOST -> DEVICE
        // --------------------------------------------------

        cudaMemcpy(
            d_input_size,
            input_size.data(),
            num_of_current_thread * sizeof(size_t),
            cudaMemcpyHostToDevice);

        // --------------------------------------------------
        // Temporary host arrays containing device pointers
        // --------------------------------------------------

        char **temp_d_input =
            new char *[num_of_current_thread];

        char **temp_d_output =
            new char *[num_of_current_thread];

        // --------------------------------------------------
        // Allocate each chunk on DEVICE
        // --------------------------------------------------

        for (int i = 0; i < num_of_current_thread; i++)
        {
            // Input
            cudaMallocTracked(
                &temp_d_input[i],
                input_size[i] * sizeof(char));

            // Output
            //
            // Currently output size == input size
            // because your kernel only copies data.
            cudaMallocTracked(
                &temp_d_output[i],
                (input_size[i] + 2) * sizeof(char));

            // --------------------------------------------------
            // Copy input chunk: HOST -> DEVICE
            // --------------------------------------------------

            cudaMemcpy(
                temp_d_input[i],
                src_file[i],
                input_size[i] * sizeof(char),
                cudaMemcpyHostToDevice);
        }

        // --------------------------------------------------
        // Copy DEVICE POINTER ARRAYS to DEVICE
        // --------------------------------------------------

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

        // --------------------------------------------------
        // Launch kernel
        // --------------------------------------------------

        int threads = MAX_THREADS_PER_BLOCK;

        int blocks =
            (num_of_current_thread + threads - 1) / threads;

        // std::cout << blocks << " " << threads << endl;

        ////////////////////paq9_cuda call////////////////////////////

        // std::cout << "Assigned block: " << blocks << endl;
        threads = std::min(threads, (int)num_of_current_thread);
        // std::cout << "Assigned threads: " << threads << endl;
        // std::cout << "Total threads: " << blocks * threads << endl;

        paq9_cuda<<<blocks, threads>>>(
            d_input_size,
            d_input,
            d_output_size,
            d_output,
            num_of_current_thread, COMPRESS, memory_level);

        cudaDeviceSynchronize();
        cudaError_t err1;
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

        finish_device_call_memory();

        // --------------------------------------------------
        // Copy output sizes: DEVICE -> HOST
        // --------------------------------------------------

        // FIX: Allocate memory for the host integer array before copying
        size_t *output_size = (size_t *)malloc(num_of_current_thread * sizeof(size_t));
        cudaMemcpy(output_size, d_output_size, num_of_current_thread * sizeof(size_t), cudaMemcpyDeviceToHost);

        // --------------------------------------------------
        // Copy output chunks: DEVICE -> HOST
        // --------------------------------------------------

        // FIX: Allocate memory for the array of host pointers before copying
        char **output = new char *[num_of_current_thread];

        for (int i = 0; i < num_of_current_thread; i++)
        {
            // FIX: Allocate memory for each specific chunk array before copying
            output[i] = new char[output_size[i]];
            cudaMemcpy(
                output[i],
                temp_d_output[i],
                output_size[i] * sizeof(char),
                cudaMemcpyDeviceToHost);
        }

        // --------------------------------------------------
        // Free DEVICE chunk memory
        // --------------------------------------------------

        for (int i = 0; i < num_of_current_thread; i++)
        {
            cudaFree(temp_d_input[i]);
            cudaFree(temp_d_output[i]);
        }

        // --------------------------------------------------
        // Free DEVICE arrays
        // --------------------------------------------------

        cudaFree(d_input);
        cudaFree(d_output);

        cudaFree(d_input_size);
        cudaFree(d_output_size);

        delete[] temp_d_input;
        delete[] temp_d_output;

        // Inside your main writing logic:

        size_t total_input = 0, total_output = 0;
        for (size_t i = 0; i < num_of_current_thread; i++)
        {
            total_input += input_size[i];
            total_output += output_size[i];
        }

        std::cout << "Uncompressed  ->  Compressed\n";

        for (size_t i = 0; i < num_of_current_thread; i++)
        {

            put4_stream((U32)(input_size[i]), dest);
            put4_stream((U32)(output_size[i]), dest);
            dest.write(output[i], output_size[i]);
            // std::cout << input_size[i] << " Byte -> " << output_size[i] << " Byte" << endl;
        }

        // std::cout << "\n\nPer Device Call:" << total_input << " Byte->" << total_output << " Byte " << endl;
        total_compressed_size += total_output;
        total_uncompressed_size += total_input;
    }
    source.close();
    dest.close();

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
    // std::cout << "Decompression is cooking......" << endl;

    total_compressed_size = 0;
    total_uncompressed_size = 0;
    // constexpr size_t MB = 1024 * 1024;

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

        // std::cout << "Uncompressed to file: " << filename << endl;
        destination_file = filename.c_str();
    }
    else
    {
        // std::cout << filename << " -> " << destination_file << endl;
    }

    char mode = source.get();
    if (mode == 's')
    {
    }
    else if (mode == 'c')
    {
        maximum_heap_limit = getMaximumHeapLimit();
        maximum_free_memory = getMaximumFreeMemory();
        size_t maximum_memory = std::min(maximum_free_memory, maximum_heap_limit);
        maximum_memory = 5 * maximum_memory / 10;

        cudaDeviceSetLimit(cudaLimitMallocHeapSize, maximum_memory);
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

        size_t usize = get8_stream(source); // uncompressed total size
        memory_chunk_level = get4_stream(source);
        memory_level = get4_stream(source);
        level = get4_stream(source);
        int num_of_chunks = get4_stream(source);

        // int device_call_count = get4_stream(source);
        // int maximum_thread_per_device_call = get4_stream(source);

        size_t memory_per_thread = 3.5 * getBytesFromMemoryLevel(memory_level);

        int maximum_thread_per_device_call = (maximum_memory + memory_per_thread - 1) / memory_per_thread;
        int device_call_count = (num_of_chunks + maximum_thread_per_device_call - 1) / maximum_thread_per_device_call;
        // size_t chunk_B = memory_chunk_level * MB;
        std::cout << "Memory Chunk Level: " << memory_chunk_level << "MB" << endl;
        std::cout << "Memory Level: " << memory_level << endl;
        std::cout << "Level: " << level << endl;
        std::cout << "Number of Chunks: " << num_of_chunks << endl;
        std::cout << "Total threads: " << num_of_chunks << endl;
        std::cout << "Maximum Thread at a time: " << maximum_thread_per_device_call << endl;
        // std::cout << "Maximum processed per device call: "
        //           << maximum_thread_per_device_call * chunk_B / MB << " MB" << endl;
        // std::cout << "Total Device Call " << device_call_count << endl;

        // device initialization
        init<<<1, 1>>>(memory_level);
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

        // output file configuration
        std::ofstream dest(destination_file, std::ios::binary);
        if (!dest)
        {
            std::cout << std::string(destination_file) << " does not created/opened.\n";
            exit(1);
        }
        for (int call_count = 0; call_count < device_call_count; call_count++)
        {
            // std::cout << "\n\nDevice Call No: " << call_count + 1 << endl;
            int num_of_current_thread = std::min(maximum_thread_per_device_call, (num_of_chunks - call_count * maximum_thread_per_device_call));

            std::vector<char *> input(num_of_current_thread);
            std::vector<size_t> input_size(num_of_current_thread);
            std::vector<size_t> uncompressed_size(num_of_current_thread);
            for (int i = 0; i < num_of_current_thread; i++)
            {
                uncompressed_size[i] = get4_stream(source);
                input_size[i] = get4_stream(source);
                input[i] = get_input(source, input_size[i]);
            }

            // preparing for calling device function

            // --------------------------------------------------
            // Device pointer arrays
            // --------------------------------------------------

            char **d_input;
            char **d_output;

            begin_device_call_memory();
            cudaMallocTracked(&d_input, num_of_current_thread * sizeof(char *));
            cudaMallocTracked(&d_output, num_of_current_thread * sizeof(char *));

            // --------------------------------------------------
            // Size arrays
            // --------------------------------------------------

            size_t *d_input_size;
            size_t *d_output_size;

            cudaMallocTracked(&d_input_size,
                              num_of_current_thread * sizeof(size_t));

            cudaMallocTracked(&d_output_size, num_of_current_thread * sizeof(size_t));
            // --------------------------------------------------
            // Copy input sizes: HOST -> DEVICE
            // --------------------------------------------------

            cudaMemcpy(
                d_input_size,
                input_size.data(),
                num_of_current_thread * sizeof(size_t),
                cudaMemcpyHostToDevice);

            // --------------------------------------------------
            // Temporary host arrays containing device pointers
            // --------------------------------------------------

            char **temp_d_input =
                new char *[num_of_current_thread];

            char **temp_d_output =
                new char *[num_of_current_thread];

            // --------------------------------------------------
            // Allocate each chunk on DEVICE
            // --------------------------------------------------

            for (int i = 0; i < num_of_current_thread; i++)
            {
                // Input
                cudaMallocTracked(
                    &temp_d_input[i],
                    input_size[i] * sizeof(char));

                // Output
                //
                // Currently output size == input size
                // because your kernel only copies data.
                cudaMallocTracked(
                    &temp_d_output[i],
                    (uncompressed_size[i]) * sizeof(char));

                // --------------------------------------------------
                // Copy input chunk: HOST -> DEVICE
                // --------------------------------------------------

                cudaMemcpy(
                    temp_d_input[i],
                    input[i],
                    input_size[i] * sizeof(char),
                    cudaMemcpyHostToDevice);
            }

            // --------------------------------------------------
            // Copy DEVICE POINTER ARRAYS to DEVICE
            // --------------------------------------------------

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

            // --------------------------------------------------
            // Launch kernel
            // --------------------------------------------------

            int threads = MAX_THREADS_PER_BLOCK;

            int blocks =
                (num_of_current_thread + threads - 1) / threads;

            // std::cout << blocks << " " << threads << endl;

            // initialize the gpu classes

            ////////////////////paq9_cuda call///////////////////////////

            // std::cout << "Assigned block: " << blocks << endl;
            threads = std::min(threads, (int)num_of_current_thread);
            // std::cout << "Assigned threads: " << threads << endl;
            // std::cout << "Total threads: " << blocks * threads << endl;

            paq9_cuda<<<blocks, threads>>>(
                d_input_size,
                d_input,
                d_output_size,
                d_output,
                num_of_current_thread, DECOMPRESS, memory_level);

            cudaDeviceSynchronize();

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

            finish_device_call_memory();

            // --------------------------------------------------
            // Copy output sizes: DEVICE -> HOST
            // --------------------------------------------------

            // FIX: Allocate memory for the host integer array before copying
            size_t *output_size = (size_t *)malloc(num_of_current_thread * sizeof(size_t));
            cudaMemcpy(output_size, d_output_size, num_of_current_thread * sizeof(size_t), cudaMemcpyDeviceToHost);

            // --------------------------------------------------
            // Copy output chunks: DEVICE -> HOST
            // --------------------------------------------------

            // FIX: Allocate memory for the array of host pointers before copying
            char **output = new char *[num_of_current_thread];

            for (int i = 0; i < num_of_current_thread; i++)
            {
                // FIX: Allocate memory for each specific chunk array before copying
                output[i] = new char[output_size[i]];
                cudaMemcpy(
                    output[i],
                    temp_d_output[i],
                    output_size[i] * sizeof(char),
                    cudaMemcpyDeviceToHost);
            }

            // --------------------------------------------------
            // Free DEVICE chunk memory
            // --------------------------------------------------

            for (int i = 0; i < num_of_current_thread; i++)
            {
                cudaFree(temp_d_input[i]);
                cudaFree(temp_d_output[i]);
            }

            // --------------------------------------------------
            // Free DEVICE arrays
            // --------------------------------------------------

            cudaFree(d_input);
            cudaFree(d_output);

            cudaFree(d_input_size);
            cudaFree(d_output_size);

            delete[] temp_d_input;
            delete[] temp_d_output;

            // Inside your main writing logic:

            size_t total_input = 0, total_output = 0;
            std::cout << "Compressed  ->  Decompressed \n";
            for (size_t i = 0; i < num_of_current_thread; i++)
            {

                dest.write(output[i], output_size[i]);
                total_input += input_size[i];
                total_output += output_size[i];
                // std::cout << input_size[i] << " Byte -> " << output_size[i] << " Byte" << endl;
            }
            // std::cout << "From Device Call: " << total_input << " Byte -> " << total_output << " Byte" << endl;
            total_compressed_size += total_input;
            total_uncompressed_size += total_output;
        }
        source.close();
        dest.close();
        std::cout << "Total: " << total_compressed_size << " Byte -> "
                  << total_uncompressed_size << " Byte" << endl;
    }
    else
    {
        std::cout << "Run again and provide proper arguments.\n";
        exit(1);
    }
}
int main(int argc, char **args)
{

    auto start = std::chrono::steady_clock::now();
    std::cout << "CUDA version of PAQ9 started successfully.\n\n";
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
    std::cout << "Working mode: "
              << (mode == COMPRESS ? "Compressing" : "Decompressing")
              << endl;
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
            for (int i = 1; i < len; i++)
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

    cudaDeviceSynchronize(); // GPU কাজ শেষ হওয়া নিশ্চিত

    std::cout << "Total GPU memory allocated: "
              << maximum_device_call_memory << " bytes ("
              << static_cast<double>(maximum_device_call_memory) / (1024 * 1024)
              << " MiB)" << endl;
    std::cout << "  cudaMalloc: " << maximum_cuda_malloc_for_device_call
              << " bytes; device heap: " << maximum_device_heap_for_device_call
              << " bytes" << endl;

    auto end = std::chrono::steady_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Time Taken: "
              << seconds << " seconds\n";
    std::cout << "Compression/Decompression Speed: " << total_uncompressed_size / seconds / 1024 << " KB/seconds \n";

    return 0;
}
