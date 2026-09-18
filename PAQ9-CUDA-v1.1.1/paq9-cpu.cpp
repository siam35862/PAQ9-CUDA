// paq9_cpu.cpp
//
// CPU multithreaded port of paq9-cuda.cu.
//
// This is a line-by-line port of the CUDA PAQ9 compressor to plain,
// portable C++ using std::thread instead of CUDA kernels/blocks/threads.
//
// Design goals (in order of priority):
//   1. Bit-exact correctness: every integer operation in the modeling
//      pipeline (Squash, Stretch, StateMap, Mix, APM, HashTable, LZP,
//      Predictor, Encoder) is copied unchanged from the .cu source, with
//      only __device__ qualifiers and CUDA-specific plumbing (tid-indexed
//      global arrays, cudaMalloc, kernels) removed. No numeric behavior is
//      changed.
//   2. File-format compatibility: the archive header and per-chunk layout
//      written by compress()/read by decompress() are byte-for-byte the
//      same as the CUDA version ("PAQ9-CUDA" magic, version, filename,
//      mode byte, sizes, per chunk 4+4 byte size fields, then the raw
//      compressed chunk bytes). Since every chunk is modeled completely
//      independently (models are re-initialized from scratch per chunk,
//      exactly like the CUDA version re-inits per device call), any file
//      produced by the CUDA build can be decompressed by this CPU build
//      and vice versa, and either build can decompress files produced by
//      the other's compressor with any thread count.
//   3. CPU multithreading: instead of one CUDA thread per chunk, one
//      std::thread worker handles a batch of chunks; workers reuse
//      persistent per-slot scratch buffers (mirroring the CUDA
//      ThreadBuffers reuse across device calls).
//
// Build:
//   g++ -O3 -std=c++17 -pthread paq9_cpu.cpp -o paq9cpu
//
// Usage (identical to the CUDA build):
//   paq9cpu -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>
//   paq9cpu -d <source_file> [<destination_file>]

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>


#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

#define COMPRESS 0
#define DECOMPRESS 1
#define endl std::endl
#define Base_Memory_Level 19
constexpr size_t MB = 1024ULL * 1024ULL;

int memory_level = 1; // default memory level MEM=1<<19+memory_level
int chunk_MB = 1; // chunk size in MB
int chunk_level = 1;

std::atomic<size_t> total_uncompressed_size{0};
std::atomic<size_t> total_compressed_size{0};

// Global memory-size parameter shared by all chunk workers. It is set once
// (single-threaded) before any worker threads are spawned for a given
// compress()/decompress() call, and only read afterward, so sharing it as
// a plain global is safe.
static size_t g_MEM = 1ULL << (Base_Memory_Level + 1);

///////////////////////////// Squash //////////////////////////////

// return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
class Squash
{
    short tab[4096];

public:
    Squash();
    int operator()(int d) const;
};

Squash::Squash() {
    static const int t[33] = {1,    2,    3,    6,    10,   16,   27,   45,   73,   120,  194,
                              310,  488,  747,  1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785,
                              3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094};
    for (int i = -2048; i < 2048; ++i) {
        int w = i & 127;
        int d = (i >> 7) + 16;
        tab[i + 2048] = (t[d] * (128 - w) + t[(d + 1)] * w + 64) >> 7;
    }
}
int Squash::operator()(int d) const {
    d += 2048;
    if (d < 0)
        return 0;
    else if (d > 4095)
        return 4095;
    else
        return tab[d];
}

//////////////////////////// Stretch ///////////////////////////////

class Stretch
{
    short t[4096];

public:
    explicit Stretch(const Squash& squash);
    int operator()(int p) const;
};

Stretch::Stretch(const Squash& squash) {
    int pi = 0;
    for (int x = -2047; x <= 2047; ++x) { // invert squash()
        int i = squash(x);
        for (int j = pi; j <= i; ++j)
            t[j] = x;
        pi = i + 1;
    }
    t[4095] = 2047;
}

int Stretch::operator()(int p) const {
    assert(p >= 0 && p < 4096);
    return t[p];
}

// Global, read-only after construction; shared safely across threads.
static Squash g_squash;
static Stretch g_stretch(g_squash);

///////////////////////// state table ////////////////////////

static const U8 State_table[256][2] = {
    {1, 2},     {3, 5},     {4, 6},     {7, 10},    {8, 12},    {9, 13},    {11, 14},   {15, 19},   {16, 23},
    {17, 24},   {18, 25},   {20, 27},   {21, 28},   {22, 29},   {26, 30},   {31, 33},   {32, 35},   {32, 35},
    {32, 35},   {32, 35},   {34, 37},   {34, 37},   {34, 37},   {34, 37},   {34, 37},   {34, 37},   {36, 39},
    {36, 39},   {36, 39},   {36, 39},   {38, 40},   {41, 43},   {42, 45},   {42, 45},   {44, 47},   {44, 47},
    {46, 49},   {46, 49},   {48, 51},   {48, 51},   {50, 52},   {53, 43},   {54, 57},   {54, 57},   {56, 59},
    {56, 59},   {58, 61},   {58, 61},   {60, 63},   {60, 63},   {62, 65},   {62, 65},   {50, 66},   {67, 55},
    {68, 57},   {68, 57},   {70, 73},   {70, 73},   {72, 75},   {72, 75},   {74, 77},   {74, 77},   {76, 79},
    {76, 79},   {62, 81},   {62, 81},   {64, 82},   {83, 69},   {84, 71},   {84, 71},   {86, 73},   {86, 73},
    {44, 59},   {44, 59},   {58, 61},   {58, 61},   {60, 49},   {60, 49},   {76, 89},   {76, 89},   {78, 91},
    {78, 91},   {80, 92},   {93, 69},   {94, 87},   {94, 87},   {96, 45},   {96, 45},   {48, 99},   {48, 99},
    {88, 101},  {88, 101},  {80, 102},  {103, 69},  {104, 87},  {104, 87},  {106, 57},  {106, 57},  {62, 109},
    {62, 109},  {88, 111},  {88, 111},  {80, 112},  {113, 85},  {114, 87},  {114, 87},  {116, 57},  {116, 57},
    {62, 119},  {62, 119},  {88, 121},  {88, 121},  {90, 122},  {123, 85},  {124, 97},  {124, 97},  {126, 57},
    {126, 57},  {62, 129},  {62, 129},  {98, 131},  {98, 131},  {90, 132},  {133, 85},  {134, 97},  {134, 97},
    {136, 57},  {136, 57},  {62, 139},  {62, 139},  {98, 141},  {98, 141},  {90, 142},  {143, 95},  {144, 97},
    {144, 97},  {68, 57},   {68, 57},   {62, 81},   {62, 81},   {98, 147},  {98, 147},  {100, 148}, {149, 95},
    {150, 107}, {150, 107}, {108, 151}, {108, 151}, {100, 152}, {153, 95},  {154, 107}, {108, 155}, {100, 156},
    {157, 95},  {158, 107}, {108, 159}, {100, 160}, {161, 105}, {162, 107}, {108, 163}, {110, 164}, {165, 105},
    {166, 117}, {118, 167}, {110, 168}, {169, 105}, {170, 117}, {118, 171}, {110, 172}, {173, 105}, {174, 117},
    {118, 175}, {110, 176}, {177, 105}, {178, 117}, {118, 179}, {110, 180}, {181, 115}, {182, 117}, {118, 183},
    {120, 184}, {185, 115}, {186, 127}, {128, 187}, {120, 188}, {189, 115}, {190, 127}, {128, 191}, {120, 192},
    {193, 115}, {194, 127}, {128, 195}, {120, 196}, {197, 115}, {198, 127}, {128, 199}, {120, 200}, {201, 115},
    {202, 127}, {128, 203}, {120, 204}, {205, 115}, {206, 127}, {128, 207}, {120, 208}, {209, 125}, {210, 127},
    {128, 211}, {130, 212}, {213, 125}, {214, 137}, {138, 215}, {130, 216}, {217, 125}, {218, 137}, {138, 219},
    {130, 220}, {221, 125}, {222, 137}, {138, 223}, {130, 224}, {225, 125}, {226, 137}, {138, 227}, {130, 228},
    {229, 125}, {230, 137}, {138, 231}, {130, 232}, {233, 125}, {234, 137}, {138, 235}, {130, 236}, {237, 125},
    {238, 137}, {138, 239}, {130, 240}, {241, 125}, {242, 137}, {138, 243}, {130, 244}, {245, 135}, {246, 137},
    {138, 247}, {140, 248}, {249, 135}, {250, 69},  {80, 251},  {140, 252}, {249, 135}, {250, 69},  {80, 251},
    {140, 252}, {0, 0},     {0, 0},     {0, 0}};

#define nex(state, sel) State_table[state][sel]

//////////////////////////// StateMap //////////////////////////

static int state_map_dt[1024];
static void init_state_map_dt() {
    for (int i = 0; i < 1024; i++)
        state_map_dt[i] = 16384 / (i + i + 3);
}

class StateMap
{
protected:
    const int N;
    int cntxt;
    U32* prediction_table; // cntxt -> prediction in high 22 bits, count in low 10 bits

public:
    StateMap(U32* prediction_table_ptr, int n = 256);

    void update(int y, int limit = 255);
    int predict_next_bit(int cntx);
};

StateMap::StateMap(U32* prediction_table_ptr, int n) : N(n), cntxt(0), prediction_table(prediction_table_ptr) {
    for (int i = 0; i < N; i++)
        prediction_table[i] = 2147483648U; // 1<<31
}

void StateMap::update(int y, int limit) {
    assert(cntxt >= 0 && cntxt < N);
    int n = prediction_table[cntxt] & 1023, p = prediction_table[cntxt] >> 10; // count, prediction

    if (n < limit)
        prediction_table[cntxt]++;
    else
        prediction_table[cntxt] = prediction_table[cntxt] & 0xfffffc00 | limit;

    prediction_table[cntxt] += (((y << 22) - p) >> 3) * state_map_dt[n] & 0xfffffc00;
}

int StateMap::predict_next_bit(int cntx) {
    assert(cntx >= 0 && cntx < N);
    return prediction_table[cntxt = cntx] >> 20;
}

//////////////////////////// Mix, APM /////////////////////////

class Mix
{
protected:
    const int N;
    int* wt;
    int x1, x2;
    int context;
    int last_prediction;

public:
    Mix(int* weight_ptr, int n = 512);
    int prediction(int p1, int p2, int cntxt);
    void update(int y);
};

Mix::Mix(int* weight_ptr, int n) : N(n), wt(weight_ptr), x1(0), x2(0), context(0), last_prediction(0) {
    for (int i = 0; i < N * 2; i++)
        wt[i] = 1 << 23;
}

int Mix::prediction(int p1, int p2, int cntxt) {
    assert(cntxt >= 0 & cntxt < N);
    context = cntxt * 2;
    return last_prediction = ((x1 = p1) * (wt[context] >> 16) + (x2 = p2) * (wt[context + 1] >> 16) + 128) >> 8;
}

void Mix::update(int y) {
    assert(y == 0 || y == 1);
    int error = ((y << 12) - g_squash(last_prediction));
    if ((wt[context] & 3) < 3) {
        error *= 4 - (++wt[context] & 3);
    }
    error = (error + 8) >> 4;
    wt[context] += x1 * error & -4;
    wt[context + 1] += x2 * error;
}

class APM : public Mix
{
public:
    APM(int* weight_ptr, int n);
};

APM::APM(int* weight_ptr, int n) : Mix(weight_ptr, n) {
    for (int i = 0; i < n; i++) {
        wt[2 * i] = 0;
    }
}

//////////////////////////// HashTable /////////////////////////

template <int B>
class HashTable
{
    U8* table; // table: 1 element = B bytes: checksum priority data
    const U32 N;

public:
    HashTable(U32 n, U8* table_ptr);
    U8* operator[](U32 i);
};

template <int B>
HashTable<B>::HashTable(U32 n, U8* table_ptr) : table(table_ptr), N(n) {
    assert(B >= 2 && (B & (B - 1)) == 0);
    assert(N >= (U32)(B * 4) && (N & (N - 1)) == 0);
    table += 64 - int(reinterpret_cast<uintptr_t>(table) & 63); // align on cache line boundary
}

template <int B>
U8* HashTable<B>::operator[](U32 i) {
    i *= 123456791;
    i = i << 16 | i >> 16;
    i *= 234567891;
    int chk = i >> 24;
    i = i * B & (N - B);
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

static inline bool isalpha_cpu(char ch) { return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'); }

static inline char tolower_cpu(char ch) {
    if (ch >= 'A' && ch <= 'Z')
        ch += 'a' - 'A';
    return ch;
}

// LZP predicts the next byte and maintains context.  Methods:
// predict_char() returns the predicted byte for the next update, or -1 if none.
// probability() returns the 12 bit probability (0..4095) that predict_char() is next.
// update(ch) updates the model with actual byte ch (0..255).
// context(i) returns the i'th prior byte of context, i > 0.
// context4() returns the order 4 context, shifted into the LSB.
// context8() returns a hash of the order 8 context, shifted 4 bits into LSB.
// word0, word1 are hashes of the current and previous word (a-z).
class LZP
{
private:
    const size_t N, H; // buffer, table size
    enum
    {
        MINLEN = 12
    }; // minimum match length
    U8* buffer; // Rotating buffer of size N
    U32* table; // Hash Table of pointers in high 24 bits, state in low 8 bits
    int match; // start of match
    size_t len; // length of match
    size_t pos; // position of next char to write to buffer
    U32 hash; // context hash
    U32 hash1; // hash of last 8 bytes updates, shifting 4 bits to MSB
    U32 hash2; // last 4 updates, shifting 8 bits to MSB
    StateMap* statemap; // len+offset->p
    APM *apm1, *apm2, *apm3; // p, context->p
    int literals, matches; // statistics
public:
    U32 word0, word1; // Hashes of last 2 words (case insensitive a-z)
    LZP(StateMap* statemap1, U8* buffer, U32* table, APM* apm1, APM* apm2, APM* apm3);
    int predict_char();
    int context(int i);
    int context4() { return hash2; }
    int context8() { return hash1; }
    int probability();
    void update(int ch);
};

LZP::LZP(StateMap* statemap, U8* buf, U32* tab, APM* a1, APM* a2, APM* a3) :
    N(g_MEM / 8), H(g_MEM / 32), match(-1), len(0), pos(0), hash(0), hash1(0), hash2(0), statemap(statemap), apm1(a1),
    apm2(a2), apm3(a3), literals(0), matches(0), word0(0), word1(0) {
    assert(g_MEM > 0);
    assert(H > 0);
    buffer = buf;
    table = tab;
}

int LZP::predict_char() { return len >= MINLEN ? buffer[match & (N - 1)] : -1; }

int LZP::context(int i) {
    assert(i > 0);
    return buffer[(pos - i) & (N - 1)];
}

int LZP::probability() {
    if (len < MINLEN)
        return 0;
    int cxt = static_cast<int>(len);
    if (len > 28)
        cxt = 28 + (len >= 32) + (len >= 64) + (len >= 128);
    int pc = predict_char();
    int pr = statemap->predict_next_bit(cxt);
    pr = g_stretch(pr);
    pr = apm1->prediction(2048, pr * 2, (hash2 * 256 + pc) & 0xffff) * 3 + pr >> 2;
    pr = apm2->prediction(2048, pr * 2, (hash1 * (11 << 6) + pc) & 0x3ffff) * 3 + pr >> 2;
    pr = apm3->prediction(2048, pr * 2, (hash1 * (7 << 4) + pc) & 0xfffff) * 3 + pr >> 2;
    pr = g_squash(pr);
    return pr;
}

void LZP::update(int ch) {
    int y = predict_char() == ch; // 1 if prediction of ch was right, else 0
    hash1 = hash1 * (3 << 4) + ch + 1; // update context hashes
    hash2 = hash2 << 8 | ch;
    hash = (hash * (5 << 2) + ch + 1) & (H - 1);
    if (len >= MINLEN) {
        statemap->update(y);
        apm1->update(y);
        apm2->update(y);
        apm3->update(y);
    }
    if (isalpha_cpu((char)ch))
        word0 = word0 * (29 << 2) + tolower_cpu((char)ch);
    else if (word0)
        word1 = word0, word0 = 0;
    buffer[pos & (N - 1)] = (U8)ch; // update buffer
    ++pos;
    if (y) { // extend match
        ++len;
        ++match;
        ++matches;
    }
    else { // find new match, try order 6 context first
        ++literals;
        y = 0;
        len = 1;
        match = table[hash];
        if (!((match ^ pos) & (N - 1)))
            --match;
        while (len <= 128 && buffer[(match - len) & (N - 1)] == buffer[(pos - len) & (N - 1)])
            ++len;
        --len;
    }
    table[hash] = (U32)pos;
}

//////////////////////////// Predictor /////////////////////////

// A Predictor estimates the probability that the next bit of
// uncompressed data is 1.
class Predictor
{
    enum
    {
        N = 11
    }; // number of contexts
    int c0; // last 0-7 bits with leading 1, 0 before LZP flag
    int nibble; // last 0-3 bits with leading 1 (1..15)
    int bcount; // number of bits in c0 (0..7)
    HashTable<16>* hashtable; // context -> state
    StateMap* statemap[N]; // state -> prediction, N size
    U8* cp[N]; // i -> state array of bit histories for i'th context
    U8* sp[N]; // i -> pointer to bit history for i'th context
    Mix* mix[N - 1]; // combines 2 predictions given a context
    APM *apm1, *apm2, *apm3; // adjusts a prediction given a context
    U8* context1; // order 1 contexts -> state
    LZP* lzp; // LZP instance for this chunk (replaces tid-indexed global)

public:
    Predictor(U8* context1_ptr, StateMap* statemap1[N], Mix* mix1[N - 1], APM* apm1, APM* apm2, APM* apm3,
              HashTable<16>* hashtable_ptr, LZP* lzp_ptr);
    int predict_next_bit();
    void update(int y);
};

Predictor::Predictor(U8* context1_ptr, StateMap* statemap1[N], Mix* mix1[N - 1], APM* a1, APM* a2, APM* a3,
                     HashTable<16>* hashtable_ptr, LZP* lzp_ptr) :
    c0(0), nibble(1), bcount(0), hashtable(hashtable_ptr), apm1(a1), apm2(a2), apm3(a3), context1(context1_ptr),
    lzp(lzp_ptr) {
    for (int i = 0; i < N; ++i) {
        sp[i] = cp[i] = context1;
        statemap[i] = statemap1[i];
        if (i < N - 1)
            mix[i] = mix1[i];
    }
}

void Predictor::update(int y) {
    assert(y == 0 || y == 1);
    assert(bcount >= 0 && bcount < 8);
    assert(c0 >= 0 && c0 < 256);
    assert(nibble >= 1 && nibble <= 15);
    if (c0 == 0)
        c0 = 1 - y;
    else {
        *sp[0] = nex(*sp[0], y);
        statemap[0]->update(y);
        for (int i = 1; i < N; ++i) {
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
}

int Predictor::predict_next_bit() {
    assert(lzp);
    if (c0 == 0)
        return lzp->probability();
    else {
        int pc = lzp->predict_char(); // mispredicted byte
        int r = (pc + 256 >> 8 - bcount) == c0; // c0 consistent with mispredicted byte?
        U32 c4 = lzp->context4(); // last 4 whole context bytes, shifted into LSB
        U32 c8 = (lzp->context8() << 4) - 1; // hash of last 7 bytes with 4 trailing 1 bits
        if ((bcount & 3) == 0) { // nibble boundary?  Update context pointers
            pc &= -r;
            U32 c4p = c4 << 8;
            if (bcount == 0) { // byte boundary?  Update order-1 context pointers
                cp[0] = context1 + (c4 >> 16 & 0xff00);
                cp[1] = context1 + (c4 >> 8 & 0xff00) + 0x10000;
                cp[2] = context1 + (c4 & 0xff00) + 0x20000;
                cp[3] = context1 + (c4 << 8 & 0xff00) + 0x30000;
            }
            cp[4] = (*hashtable)[(c4p & 0xffff00) - c0];
            cp[5] = (*hashtable)[(c4p & 0xffffff00) * 3 + c0];
            cp[6] = (*hashtable)[c4 * 7 + c0];
            cp[7] = (*hashtable)[(c8 * 5 & 0xfffffc) + c0];
            cp[8] = (*hashtable)[(c8 * 11 & 0xffffff0) + c0 + pc * 13];
            cp[9] = (*hashtable)[(lzp->word0 * 5 + c0 + pc * 17)];
            cp[10] = (*hashtable)[(lzp->word1 * 7 + lzp->word0 * 11 + c0 + pc * 37)];
        }

        // Mix predictions
        r <<= 8;
        sp[0] = &cp[0][c0];
        int pr = g_stretch(statemap[0]->predict_next_bit(*sp[0]));
        for (int i = 1; i < N; ++i) {
            sp[i] = &cp[i][i < 4 ? c0 : nibble];
            int st = *sp[i];
            pr = mix[i - 1]->prediction(pr, g_stretch(statemap[i]->predict_next_bit(st)), st + r) * 3 + pr >> 2;
        }
        pr = apm1->prediction(512, pr * 2, (c0 + pc * 256) & 0xffff) * 3 + pr >> 2; // Adjust prediction
        pr = apm2->prediction(512, pr * 2, (c4 << 8 & 0xff00) | c0) * 3 + pr >> 2;
        pr = apm3->prediction(512, pr * 2, (c4 * 3 + c0) & 0xffff) * 3 + pr >> 2;
        return g_squash(pr);
    }
}

//////////////////////////// Encoder ////////////////////////////

// An Encoder arithmetic codes in blocks of size BUFSIZE.
class Encoder
{
private:
    const int mode;
    char* inout;
    size_t total_size;
    Predictor* pred; // replaces tid-indexed global predictor[tid]

    U32 x1, x2;
    U32 x;
    enum
    {
        BUFSIZE = 0x20000
    };
    U8* buffer;
    size_t usize, csize;
    double usum, csum;

public:
    size_t iterator_size;
    Encoder(int m, char* temp, unsigned char* buffer_ptr, size_t tsz, size_t itr, Predictor* pred);
    bool flush();
    bool put4(U32 c);

    int code(int y = 0) {
        int p = pred->predict_next_bit();
        assert(p >= 0 && p < 4096);
        p += p < 2048;
        U32 xmid = x1 + (x2 - x1 >> 12) * p + ((x2 - x1 & 0xfff) * p >> 12);
        assert(xmid >= x1 && xmid < x2);
        if (mode == DECOMPRESS)
            y = x <= xmid;
        y ? (x2 = xmid) : (x1 = xmid + 1);
        pred->update(y);
        while (((x1 ^ x2) & 0xff000000) == 0) { // pass equal leading bytes of range
            if (mode == COMPRESS)
                buffer[csize++] = x2 >> 24;
            x1 <<= 8;
            x2 = (x2 << 8) + 255;
            if (mode == DECOMPRESS)
                x = (x << 8) + (unsigned char)(inout[iterator_size++]);
        }
        return y;
    }

    bool count() {
        assert(mode == COMPRESS);
        ++usize;
        if (csize > BUFSIZE - 256)
            return flush();
        return true;
    }
};

Encoder::Encoder(int m, char* temp, unsigned char* buffer_ptr, size_t tsz, size_t itr, Predictor* pr) :
    mode(m), inout(temp), total_size(tsz), pred(pr), x1(0), x2(0xffffffff), x(0), buffer(buffer_ptr), usize(0),
    csize(0), usum(0), csum(0), iterator_size(itr) {
    if (mode == DECOMPRESS) { // x = first 4 bytes of archive
        for (int i = 0; i < 4; ++i)
            x = (x << 8) + (unsigned char)(inout[iterator_size++]);
        csize = 4;
    }
}

bool Encoder::put4(U32 c) {
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

bool Encoder::flush() {
    if (mode == COMPRESS) {
        buffer[csize++] = x1 >> 24;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        buffer[csize++] = 255;
        if (!put4((U32)usize))
            return false;
        if (!put4((U32)csize))
            return false;
        for (size_t i = 0; i < csize; i++) {
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

static size_t get4(size_t& itr, const char* in) {
    size_t r = (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    r = r * 256 + (unsigned char)in[itr++];
    return r;
}

//////////////////////// Per-chunk scratch buffers ////////////////////////
//
// Mirrors CUDA's ThreadBuffers: one persistent set of raw scratch arrays
// per worker slot, reused across chunks. Sizes match the .cu allocation
// sizes exactly (see calculateThreadBufferBytes below), sized from the
// process-wide g_MEM (set from memory_level before any chunk work starts).

struct ChunkBuffers
{
    std::vector<U32> lzp_statemap; // 0x200
    std::vector<int> lzp_apm0; // 0x20000  (APM n=0x10000)
    std::vector<int> lzp_apm1; // 0x80000  (APM n=0x40000)
    std::vector<int> lzp_apm2; // 0x200000 (APM n=0x100000)
    std::vector<U8> lzp_buffer; // g_MEM/8
    std::vector<U32> lzp_table; // g_MEM/32

    std::vector<U32> predictor_statemap[11]; // each 0x100
    std::vector<int> predictor_mix[10]; // each 0x400 (Mix n=0x200)
    std::vector<int> predictor_apm[3]; // each 0x20000 (APM n=0x10000)
    std::vector<U8> predictor_hashtable; // g_MEM/2 + 128
    std::vector<U8> predictor_context1; // 0x40000

    std::vector<U8> encoder_buffer; // 0x20000 (Encoder::BUFSIZE)

    void init() {
        lzp_statemap.assign(0x200, 0);
        lzp_apm0.assign(0x20000, 0);
        lzp_apm1.assign(0x80000, 0);
        lzp_apm2.assign(0x200000, 0);
        lzp_buffer.assign(g_MEM / 8, 0);
        lzp_table.assign(g_MEM / 32, 0);

        for (int j = 0; j < 11; j++)
            predictor_statemap[j].assign(0x100, 0);
        for (int j = 0; j < 10; j++)
            predictor_mix[j].assign(0x400, 0);
        for (int j = 0; j < 3; j++)
            predictor_apm[j].assign(0x20000, 0);
        predictor_hashtable.assign(g_MEM / 2 + 128, 0);
        predictor_context1.assign(0x40000, 0);

        encoder_buffer.assign(0x20000, 0);
    }

    // Reset the arrays that must start zeroed for a fresh chunk's model
    // (matches the memsets in the CUDA init() kernel). The StateMap/Mix/APM
    // constructors below fully overwrite every element of their own arrays,
    // so those don't need clearing here.
    void reset() {
        std::fill(predictor_hashtable.begin(), predictor_hashtable.end(), (U8)0);
        std::fill(predictor_context1.begin(), predictor_context1.end(), (U8)0);
        std::fill(lzp_buffer.begin(), lzp_buffer.end(), (U8)0);
        std::fill(lzp_table.begin(), lzp_table.end(), (U32)0);
    }
};

// Total bytes one ChunkBuffers instance occupies for a given g_MEM. Used
// only to budget how many worker slots we can afford to keep resident.
static size_t calculateThreadBufferBytes() {
    size_t total = 0;
    total += 0x200 * sizeof(U32);
    total += 0x20000 * sizeof(int);
    total += 0x80000 * sizeof(int);
    total += 0x200000 * sizeof(int);
    total += (g_MEM / 8) * sizeof(U8);
    total += (g_MEM / 32) * sizeof(U32);

    total += 11 * (0x100 * sizeof(U32));
    total += 10 * (0x400 * sizeof(int));
    total += 3 * (0x20000 * sizeof(int));
    total += (g_MEM / 2 + 128) * sizeof(U8);
    total += 0x40000 * sizeof(U8);

    total += 0x20000 * sizeof(U8);
    return total;
}

//////////////////////// Chunk-level compress / decompress ////////////////////////
//
// Direct port of the body of the paq9_cuda<<<>>> kernel for a single chunk
// (single CUDA thread's worth of work), operating on the caller-owned
// scratch buffers `b`. Every model object here is freshly constructed on
// top of freshly-reset raw storage, exactly like the CUDA build's init()
// kernel re-initializes a thread's models before every device call.

static void compressChunk(const char* input, size_t input_size, char* output, size_t& output_size, ChunkBuffers& b) {
    b.reset();

    StateMap lzp_sm(b.lzp_statemap.data(), 0x200);
    APM lzp_apm1(b.lzp_apm0.data(), 0x10000);
    APM lzp_apm2(b.lzp_apm1.data(), 0x40000);
    APM lzp_apm3(b.lzp_apm2.data(), 0x100000);
    LZP lzp(&lzp_sm, b.lzp_buffer.data(), b.lzp_table.data(), &lzp_apm1, &lzp_apm2, &lzp_apm3);

    std::vector<StateMap> psm;
    psm.reserve(11);
    for (int j = 0; j < 11; j++)
        psm.emplace_back(b.predictor_statemap[j].data(), 0x100);
    std::vector<Mix> pmix;
    pmix.reserve(10);
    for (int j = 0; j < 10; j++)
        pmix.emplace_back(b.predictor_mix[j].data(), 0x200);
    APM predictor_apm1(b.predictor_apm[0].data(), 0x10000);
    APM predictor_apm2(b.predictor_apm[1].data(), 0x10000);
    APM predictor_apm3(b.predictor_apm[2].data(), 0x10000);
    HashTable<16> hashtable((U32)(g_MEM / 2), b.predictor_hashtable.data());

    StateMap* sm_ptrs[11];
    Mix* mix_ptrs[10];
    for (int j = 0; j < 11; j++)
        sm_ptrs[j] = &psm[j];
    for (int j = 0; j < 10; j++)
        mix_ptrs[j] = &pmix[j];

    Predictor pred(b.predictor_context1.data(), sm_ptrs, mix_ptrs, &predictor_apm1, &predictor_apm2, &predictor_apm3,
                   &hashtable, &lzp);

    // NOTE: tsz is intentionally the *original* chunk size (not the physical
    // output buffer capacity). This is what makes the encoder bail out to
    // store_mode when compression fails to shrink the chunk -- exactly as
    // in the CUDA version, where Encoder's total_size bound is input_size[tid].
    // The physical output buffer is allocated larger (chunk_B+2) by the
    // caller so writes up to this bound (plus the small, bounded overshoot
    // that a single flush() can incur before the next bound check) never
    // overrun real memory.
    size_t itr = 0;
    Encoder encoder(COMPRESS, output, b.encoder_buffer.data(), input_size, itr, &pred);
    int store_mode = 0;
    output[encoder.iterator_size++] = '0';
    itr = 0;
    while (itr < input_size) {
        int ch = (unsigned char)input[itr];
        itr++;

        int cp = lzp.predict_char();
        if (ch == cp)
            encoder.code(1);
        else
            for (int i = 8; i >= 0; --i)
                encoder.code((ch >> i) & 1);
        if (!encoder.count()) {
            store_mode = 1;
            break;
        }
        lzp.update(ch);
    }
    if (!encoder.flush())
        store_mode = 1;
    if (store_mode) {
        encoder.iterator_size = 0;
        output[encoder.iterator_size++] = '1';
        itr = 0;
        while (itr < input_size) {
            output[encoder.iterator_size++] = input[itr++];
        }
    }
    output_size = encoder.iterator_size;
}

static void decompressChunk(const char* input, size_t input_size, char* output, size_t& output_size, ChunkBuffers& b) {
    b.reset();

    if (input_size > 0 && input[0] == '1') {
        size_t itr = 0, itr2 = 1;
        while (itr2 < input_size) {
            output[itr++] = input[itr2++];
        }
        output_size = itr;
        return;
    }

    StateMap lzp_sm(b.lzp_statemap.data(), 0x200);
    APM lzp_apm1(b.lzp_apm0.data(), 0x10000);
    APM lzp_apm2(b.lzp_apm1.data(), 0x40000);
    APM lzp_apm3(b.lzp_apm2.data(), 0x100000);
    LZP lzp(&lzp_sm, b.lzp_buffer.data(), b.lzp_table.data(), &lzp_apm1, &lzp_apm2, &lzp_apm3);

    std::vector<StateMap> psm;
    psm.reserve(11);
    for (int j = 0; j < 11; j++)
        psm.emplace_back(b.predictor_statemap[j].data(), 0x100);
    std::vector<Mix> pmix;
    pmix.reserve(10);
    for (int j = 0; j < 10; j++)
        pmix.emplace_back(b.predictor_mix[j].data(), 0x200);
    APM predictor_apm1(b.predictor_apm[0].data(), 0x10000);
    APM predictor_apm2(b.predictor_apm[1].data(), 0x10000);
    APM predictor_apm3(b.predictor_apm[2].data(), 0x10000);
    HashTable<16> hashtable((U32)(g_MEM / 2), b.predictor_hashtable.data());

    StateMap* sm_ptrs[11];
    Mix* mix_ptrs[10];
    for (int j = 0; j < 11; j++)
        sm_ptrs[j] = &psm[j];
    for (int j = 0; j < 10; j++)
        mix_ptrs[j] = &pmix[j];

    Predictor pred(b.predictor_context1.data(), sm_ptrs, mix_ptrs, &predictor_apm1, &predictor_apm2, &predictor_apm3,
                   &hashtable, &lzp);

    size_t itr2 = 0;
    size_t itr = 1;
    while (itr < input_size) {
        size_t usize = get4(itr, input);
        get4(itr, input); // csize (unused; the compressed data is self-delimiting)
        Encoder encoder(DECOMPRESS, const_cast<char*>(input), b.encoder_buffer.data(), input_size, itr, &pred);

        while (usize--) {
            int cp = lzp.predict_char();
            if (encoder.code() == 0) {
                cp = 1;
                while (cp < 256)
                    cp += cp + encoder.code();
                cp &= 255;
            }
            output[itr2++] = (char)cp;
            lzp.update(cp);
        }
        itr = encoder.iterator_size;
    }
    output_size = itr2;
}

//////////////////////// Stream helpers (same format as CUDA build) ////////////////////////

static unsigned int get4_stream(std::istream& in) {
    unsigned int r = (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    r = r * 256 + (unsigned char)in.get();
    return r;
}

static void put4_stream(U32 c, std::ostream& out) {
    out.put((char)((c >> 24) & 0xFF));
    out.put((char)((c >> 16) & 0xFF));
    out.put((char)((c >> 8) & 0xFF));
    out.put((char)(c & 0xFF));
}

static size_t get8_stream(std::istream& in) {
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

static void put8_stream(size_t c, std::ostream& out) {
    out.put((char)((c >> 56) & 0xFF));
    out.put((char)((c >> 48) & 0xFF));
    out.put((char)((c >> 40) & 0xFF));
    out.put((char)((c >> 32) & 0xFF));
    out.put((char)((c >> 24) & 0xFF));
    out.put((char)((c >> 16) & 0xFF));
    out.put((char)((c >> 8) & 0xFF));
    out.put((char)(c & 0xFF));
}

//////////////////////// Host CPU/RAM budget helpers ////////////////////////

static size_t getAvailableMemoryBytes() {
#if defined(__unix__) || defined(__APPLE__)
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGE_SIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0)
        return (size_t)pages * (size_t)page_size;
#endif
#endif
    return 4ULL * 1024 * 1024 * 1024; // conservative 4 GiB fallback
}

static unsigned int getHardwareThreadCount() {
    unsigned int hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4u : hc;
}

// Decide how many worker slots (persistent ChunkBuffers) to keep resident,
// bounded by hardware concurrency, the number of chunks, and available RAM
// so a large memory_level doesn't try to allocate more scratch memory than
// the machine has. Mirrors the spirit of the CUDA build's GPU-memory-based
// thread budgeting, using system RAM instead of VRAM.
static int decideThreadCount(int num_of_chunks) {
    if (const char* env = std::getenv("PAQ9_THREADS")) {
        int t = std::atoi(env);
        if (t > 0)
            return std::min(t, std::max(1, num_of_chunks));
    }

    size_t maximum_memory = getAvailableMemoryBytes();
    maximum_memory = 6 * maximum_memory / 10; // use at most 60%, like GPU_LEVEL=6

    size_t memory_per_thread = 2ULL * chunk_MB * MB + calculateThreadBufferBytes() + 1ULL * MB;
    int max_by_memory = (int)std::max<size_t>(1, maximum_memory / memory_per_thread);

    int threads = std::min<int>(getHardwareThreadCount()-2, max_by_memory);
    threads = std::min(threads, num_of_chunks);
    if (threads < 1)
        threads = 1;
    return threads;
}

//////////////////////// Top-level compress / decompress ////////////////////////

const char* get_file_name(const char* path) {
    const char* slash_pos = strrchr(path, '/');
    if (slash_pos)
        return slash_pos + 1;

#ifdef _WIN32
    const char* backslash_pos = strrchr(path, '\\');
    if (backslash_pos)
        return backslash_pos + 1;
#endif

    return path;
}

void compress(const char* destination_file, const char* source_file) {
    std::ifstream source(source_file, std::ios::binary);
    if (!source) {
        std::cerr << "Cannot open " << source_file << endl;
        exit(1);
    }
    source.seekg(0, std::ios::end);
    size_t total_B = (size_t)source.tellg();
    source.clear();
    source.seekg(0, std::ios::beg);

    chunk_MB = (1 << (chunk_level - 1));
    g_MEM = 1ULL << (Base_Memory_Level + memory_level);

    size_t chunk_B = (size_t)chunk_MB * MB;
    int num_of_chunks = (int)((total_B + chunk_B - 1) / chunk_B);
    if (num_of_chunks == 0)
        num_of_chunks = 1; // still emit a valid (empty) archive for a 0-byte input

    std::ofstream dest(destination_file, std::ios::binary);
    if (!dest) {
        std::cout << std::string(destination_file) << " does not created/opened.\n";
        exit(1);
    }

    dest.write("PAQ9-CUDA", 9); // magic (kept for cross-compatibility with the CUDA build)
    dest.put(1); // program version
    dest.write(source_file, (std::streamsize)strlen(source_file));
    dest.put(0);
    dest.put('c');
    put8_stream(total_B, dest);
    put4_stream((U32)chunk_MB, dest);
    put4_stream((U32)memory_level, dest);
    put4_stream((U32)chunk_level, dest);
    put4_stream((U32)num_of_chunks, dest);

    int num_threads = decideThreadCount(num_of_chunks);

    std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
    std::cout << "Memory Level: " << memory_level << endl;
    std::cout << "Level: " << chunk_level << endl;
    std::cout << "Number of Chunks: " << num_of_chunks << endl;
    std::cout << "Worker Threads: " << num_threads << endl;

    std::vector<ChunkBuffers> slots(num_threads);
    for (auto& s : slots)
        s.init();

    std::vector<std::vector<char>> in_buf(num_threads);
    std::vector<std::vector<char>> out_buf(num_threads);
    for (int i = 0; i < num_threads; i++) {
        in_buf[i].resize(chunk_B);
        out_buf[i].resize(chunk_B + 2);
    }

    total_uncompressed_size = 0;
    total_compressed_size = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    int chunks_done = 0;
    while (chunks_done < num_of_chunks) {
        int batch = std::min(num_threads, num_of_chunks - chunks_done);

        std::vector<size_t> in_sizes(batch), out_sizes(batch);
        for (int i = 0; i < batch; i++) {
            size_t remaining = total_B - (size_t)(chunks_done + i) * chunk_B;
            size_t cur_B = std::min(chunk_B, remaining);
            in_sizes[i] = cur_B;
            source.read(in_buf[i].data(), (std::streamsize)cur_B);
        }

        std::vector<std::thread> workers;
        workers.reserve(batch);
        for (int i = 0; i < batch; i++) {
            workers.emplace_back(
                [&, i]() { compressChunk(in_buf[i].data(), in_sizes[i], out_buf[i].data(), out_sizes[i], slots[i]); });
        }
        for (auto& t : workers)
            t.join();

        size_t batch_in = 0, batch_out = 0;
        for (int i = 0; i < batch; i++) {
            put4_stream((U32)in_sizes[i], dest);
            put4_stream((U32)out_sizes[i], dest);
            dest.write(out_buf[i].data(), (std::streamsize)out_sizes[i]);
            batch_in += in_sizes[i];
            batch_out += out_sizes[i];
        }
        total_uncompressed_size += batch_in;
        total_compressed_size += batch_out;
        std::cout << "\tTotal " << total_uncompressed_size / MB << " MB processed\n";
        auto end_time2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time2 = end_time2 - start_time;
        std::cout << "\tTotal Elapsed Time: " << elapsed_time2.count() << " second\n" << endl;

        chunks_done += batch;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total execution time: " << duration.count() << " ms" << endl;

    source.close();
    dest.close();

    std::cout << "Uncompressed  ->  Compressed\n";
    std::cout << "Total: " << total_uncompressed_size.load() << " Byte -> " << total_compressed_size.load() << " Byte"
              << endl;
    std::cout << "Compression Ratio: "
              << 1.0 * (double)total_uncompressed_size.load() /
            (double)std::max<size_t>(1, total_compressed_size.load())
              << endl;
}

bool check_archive(std::istream& in) {
    std::string magic = "PAQ9-CUDA";
    for (char c : magic) {
        if (in.get() != c)
            return false;
    }
    return in.get() == 1;
}

std::string get_file_name_stream(std::istream& in) {
    std::string file_name;
    char c;
    while (in.get(c) && c != '\0')
        file_name += c;
    return file_name;
}

char* get_input(std::istream& source, size_t size) {
    char* input = new char[size];
    source.read(input, (std::streamsize)size);
    return input;
}

void decompress(const char* destination_file, const char* source_file) {
    total_compressed_size = 0;
    total_uncompressed_size = 0;

    std::ifstream source(source_file, std::ios::binary);
    if (!source) {
        std::cerr << "Cannot open " << source_file << endl;
        exit(1);
    }

    if (!check_archive(source)) {
        std::cout << "This is not a PAQ9-CUDA compressed file.\n";
        exit(1);
    }

    std::string filename = get_file_name_stream(source);

    std::string dest_name;
    if (destination_file == 0)
        dest_name = filename;
    else
        dest_name = destination_file;

    char mode = (char)source.get();
    if (mode == 's') {
        // Reserved/unused store-mode header in the original format; nothing
        // further is defined for it there either.
    }
    else if (mode == 'c') {
        size_t usize_total = get8_stream(source); // total uncompressed size (informational)
        (void)usize_total;
        chunk_MB = (int)get4_stream(source);
        memory_level = (int)get4_stream(source);
        chunk_level = (int)get4_stream(source);
        int num_of_chunks = (int)get4_stream(source);

        g_MEM = 1ULL << (Base_Memory_Level + memory_level);
        size_t chunk_B = (size_t)chunk_MB * MB;

        std::cout << "Memory Chunk Level: " << chunk_MB << "MB" << endl;
        std::cout << "Memory Level: " << memory_level << endl;
        std::cout << "Level: " << chunk_level << endl;
        std::cout << "Number of Chunks: " << num_of_chunks << endl;

        int num_threads = decideThreadCount(num_of_chunks);
        std::cout << "Worker Threads: " << num_threads << endl;

        std::vector<ChunkBuffers> slots(num_threads);
        for (auto& s : slots)
            s.init();

        std::vector<std::vector<char>> out_buf(num_threads);
        for (int i = 0; i < num_threads; i++)
            out_buf[i].resize(chunk_B);

        std::ofstream dest(dest_name, std::ios::binary);
        if (!dest) {
            std::cout << dest_name << " does not created/opened.\n";
            exit(1);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        int chunks_done = 0;
        while (chunks_done < num_of_chunks) {
            int batch = std::min(num_threads, num_of_chunks - chunks_done);

            std::vector<std::vector<char>> in_buf(batch);
            std::vector<size_t> in_sizes(batch), out_sizes(batch);
            for (int i = 0; i < batch; i++) {
                get4_stream(source); // original uncompressed chunk size (unused: the
                                     // decoded stream is self-delimiting per block)
                size_t csize = get4_stream(source);
                in_sizes[i] = csize;
                in_buf[i].resize(csize);
                source.read(in_buf[i].data(), (std::streamsize)csize);
            }

            std::vector<std::thread> workers;
            workers.reserve(batch);
            for (int i = 0; i < batch; i++) {
                workers.emplace_back(
                    [&, i]()
                    { decompressChunk(in_buf[i].data(), in_sizes[i], out_buf[i].data(), out_sizes[i], slots[i]); });
            }
            for (auto& t : workers)
                t.join();

            size_t batch_in = 0, batch_out = 0;
            for (int i = 0; i < batch; i++) {
                dest.write(out_buf[i].data(), (std::streamsize)out_sizes[i]);
                batch_in += in_sizes[i];
                batch_out += out_sizes[i];
            }
            total_compressed_size += batch_in;
            total_uncompressed_size += batch_out;
            std::cout << "\tTotal " << total_uncompressed_size / MB << " MB decompressed\n";
            auto end_time2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time2 = end_time2 - start_time;
            std::cout << "\tTotal Elapsed Time: " << elapsed_time2.count() << " second\n" << endl;

            chunks_done += batch;
        }

        source.close();
        dest.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total execution time: " << duration.count() << " ms" << endl;
        std::cout << "Compressed  ->  Decompressed \n";
        std::cout << "Total: " << total_compressed_size.load() << " Byte -> " << total_uncompressed_size.load()
                  << " Byte" << endl;
    }
}

void print_usage(const char* prog_name) {
    const char* file_name = get_file_name(prog_name);

    std::cout << "Usage:\n";
    std::cout << "  Compress:   " << file_name
              << " -c [-<memory_level>] <destination_file> [-<chunk_level>] <source_file>\n";
    std::cout << "  Decompress: " << file_name << " -d <source_file> [<destination_file>]\n\n";

    std::cout << "  <memory_level> and <chunk_level> must be between 1 and 11.\n";
    std::cout << "  If not given, or out of bounds, both default to 1.\n\n";

    std::cout << "  memory_level: controls how much memory is used per worker thread.\n";
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

int main(int argc, char** args) {
    init_state_map_dt();

    auto start = std::chrono::steady_clock::now();
    std::cout << "CPU multithreaded PAQ9 (paq9-cuda port) started successfully.\n\n";
    if (argc < 3) {
        print_usage(args[0]);
        exit(1);
    }
    int mode;
    char* destination_file_name = 0;
    char* source_file_name = 0;
    if (args[1][0] == '-') {
        if (args[1][1] == 'c')
            mode = COMPRESS;
        else if (args[1][1] == 'd')
            mode = DECOMPRESS;
        else {
            print_usage(args[0]);
            exit(1);
        }
    }
    else {
        print_usage(args[0]);
        exit(1);
    }
    std::cout << "Working mode: " << (mode == COMPRESS ? "Compressing" : "Decompressing") << endl;
    int ind = 2;
    if (mode == COMPRESS) {
        if (ind < argc && args[ind][0] == '-') {
            std::string temp;
            size_t len = strlen(args[ind]);
            for (size_t i = 1; i < len; i++) {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            memory_level = std::stoi(temp);
            if (memory_level < 1 || memory_level > 11) {
                memory_level = 1;
                std::cout << "Your provided memory level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc) {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc) {
            destination_file_name = args[ind];
            ind++;
        }
        else {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc && args[ind][0] == '-') {
            std::string temp;
            size_t len = strlen(args[ind]);
            for (size_t i = 1; i < len; i++) {
                if (isdigit((unsigned char)args[ind][i]))
                    temp += args[ind][i];
                else {
                    print_usage(args[0]);
                    exit(1);
                }
            }
            chunk_level = std::stoi(temp);
            if (chunk_level < 1 || chunk_level > 11) {
                chunk_level = 1;
                std::cout << "Your provided level is not supported. It is set to default value 1.\n";
            }
            ind++;
        }
        else if (ind >= argc) {
            print_usage(args[0]);
            exit(1);
        }

        if (ind < argc) {
            source_file_name = args[ind];
            ind++;
        }
        else {
            print_usage(args[0]);
            exit(1);
        }

        compress(destination_file_name, source_file_name);
    }
    else {
        if (ind < argc) {
            source_file_name = args[ind];
            ind++;
        }
        else {
            print_usage(args[0]);
            exit(1);
        }
        if (ind < argc) {
            destination_file_name = args[ind];
            ind++;
        }

        decompress(destination_file_name, source_file_name);
    }

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Time Taken: " << seconds << " seconds\n";
    if (seconds > 0)
        std::cout << "Compression/Decompression Speed: " << (double)total_uncompressed_size.load() / seconds / 1024
                  << " KB/seconds \n";

    return 0;
}
