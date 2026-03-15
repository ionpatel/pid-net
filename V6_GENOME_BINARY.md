# V6: Genome Binary Format — DNA-Structured Storage

## The Principle

DNA doesn't "load." It's ALWAYS there. A transcription factor finds its target gene 
through 3D diffusion and binds in microseconds. No deserialization. No format conversion.
The storage format IS the computation format.

Our genome binary must work the same way:
- **mmap'd** — the file IS the memory (zero-copy)
- **Fixed offsets** — gene_i is at byte offset O(i), always. Pointer arithmetic = nanoseconds.
- **Cache-aligned** — genes fit in L1 cache lines (64 bytes on Apple Silicon)
- **The binary on disk = the struct in memory** — no parsing, no conversion

## Current ML (BAD)

```
Disk: safetensors file
  ↓ deserialize (milliseconds)
RAM: tensor objects  
  ↓ copy to GPU (microseconds)
GPU: compute buffers
  ↓ kernel launch (microseconds)
Result
```

**Total latency: milliseconds. Layers of abstraction. Wasteful.**

## DNA Way (GOOD)

```
Disk/RAM: mmap'd genome binary (SAME thing)
  ↓ pointer dereference (nanoseconds)
L1 cache: gene data ready
  ↓ compute (nanoseconds)
Result
```

**Total latency: nanoseconds. Zero copies. Zero parsing.**

## Binary Format: The Genome File (.genome)

```
┌──────────────────────────────────────────┐
│ HEADER (64 bytes, 1 cache line)          │
│  magic: "GENO"  (4 bytes)               │
│  version: u32                            │
│  n_chromosomes: u32                      │
│  n_genes: u32                            │
│  d_model: u32                            │
│  genome_length: u64                      │
│  epigenetic_offset: u64                  │
│  regulation_offset: u64                  │
│  checksum: u64                           │
│  padding to 64 bytes                     │
├──────────────────────────────────────────┤
│ CHROMOSOME TABLE (n_chromosomes × 64B)   │
│  Each chromosome:                        │
│    id: u32                               │
│    gene_start: u32                       │
│    gene_count: u32                       │
│    offset: u64  (byte offset in file)    │
│    size: u64                             │
│    function: u32 (enum: PERCEPTION,      │
│      MEMORY, PREDICTION, REGULATION,     │
│      FOLDING, OUTPUT)                    │
│    padding to 64 bytes                   │
├──────────────────────────────────────────┤
│ GENE TABLE (n_genes × 64 bytes)          │
│  Each gene:                              │
│    id: u32                               │
│    chromosome: u32                       │
│    offset: u64  (byte offset to data)    │
│    size: u32    (data size in bytes)     │
│    d_in: u16                             │
│    d_out: u16                            │
│    gene_type: u8 (LINEAR, OUTER_PROD,    │
│      ATTENTION, GATE, NORM, ACTIVATION)  │
│    active: u8   (epigenetic on/off)      │
│    expression_level: f16                 │
│    regulation_targets: u32[4]            │
│    padding to 64 bytes                   │
├──────────────────────────────────────────┤
│ GENE DATA (the actual weights)           │
│  Contiguous float16/bfloat16 blocks      │
│  Each gene's data at its offset          │
│  Cache-line aligned (64-byte boundary)   │
├──────────────────────────────────────────┤
│ EPIGENETIC STATE                         │
│  Methylation mask: f16[n_genes]          │
│  Accessibility: f16[n_genes]             │
│  History: rolling buffer of states       │
├──────────────────────────────────────────┤
│ REGULATION NETWORK                       │
│  Sparse adjacency: (src, dst, weight)[]  │
│  Regulatory dynamics state               │
└──────────────────────────────────────────┘
```

## C++ Access Pattern

```cpp
#include <sys/mman.h>

class Genome {
    void* base;          // mmap'd file
    Header* header;
    Chromosome* chromosomes;
    Gene* genes;
    float16_t* gene_data;
    float16_t* epigenetic;
    
public:
    Genome(const char* path) {
        int fd = open(path, O_RDWR);
        struct stat st;
        fstat(fd, &st);
        base = mmap(nullptr, st.st_size, 
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
        
        // Direct pointer arithmetic — NO parsing
        header = (Header*)base;
        chromosomes = (Chromosome*)((char*)base + 64);
        genes = (Gene*)((char*)base + 64 + header->n_chromosomes * 64);
        gene_data = (float16_t*)((char*)base + genes[0].offset);
        epigenetic = (float16_t*)((char*)base + header->epigenetic_offset);
    }
    
    // Fetch gene data in NANOSECONDS — pointer arithmetic + L1 cache
    inline float16_t* get_gene(uint32_t gene_id) {
        return (float16_t*)((char*)base + genes[gene_id].offset);
    }
    
    // Check if gene is expressed (epigenetic gate)
    inline bool is_expressed(uint32_t gene_id) {
        return epigenetic[gene_id] > genes[gene_id].expression_level;
    }
    
    // Modify epigenetic state (the genome LEARNS at runtime)
    inline void methylate(uint32_t gene_id, float16_t delta) {
        epigenetic[gene_id] += delta;
        // Changes persist to disk via mmap (the organism remembers)
    }
};
```

## Access Latency (Apple Silicon M2)

| Access | Latency | DNA Analog |
|--------|---------|------------|
| L1 cache hit | ~1 ns | Transcription factor already bound |
| L2 cache hit | ~5 ns | Gene in active chromatin |
| Unified memory | ~30 ns | Gene in accessible chromosome |
| SSD (mmap page fault) | ~50 μs | Gene in deep storage (heterochromatin) |

A 384-dim gene = 384 × 2 bytes = 768 bytes = 12 cache lines.
First access: ~30ns (RAM). Subsequent: ~1ns (L1 cached).
**One million gene lookups per millisecond.**

## Neuron Design (C++ Compute Units)

```cpp
struct Neuron {
    uint32_t gene_ids[4];     // which genes this neuron expresses
    float16_t state[384];     // current activation (neurostate)
    float16_t threshold;      // firing threshold
    uint32_t connections[16]; // outgoing synapses (sparse)
    
    // Fire only if above threshold (sparse activation = brain-like)
    inline bool should_fire() {
        float sum = 0;
        for (int i = 0; i < 384; i++) sum += state[i] * state[i];
        return sum > threshold;
    }
    
    // Process input using expressed genes
    inline void compute(Genome& genome, float16_t* input) {
        for (int g = 0; g < 4; g++) {
            if (!genome.is_expressed(gene_ids[g])) continue;
            float16_t* W = genome.get_gene(gene_ids[g]);  // NANOSECONDS
            // matmul: state = W × input (or outer product, etc.)
            matvec(W, input, state, 384);
        }
    }
};
```

## The Living Binary

The .genome file is not a checkpoint. It's a LIVING ORGANISM:
- mmap with MAP_SHARED — changes write through to disk
- Epigenetic updates persist automatically
- The file IS the brain's state at all times
- Kill the process, restart it — the organism wakes up where it left off
- Copy the .genome file = clone the organism

## Questions to Resolve

1. **float16 vs bfloat16 vs int8?** Brain uses ~1-bit spikes. Start with f16, move to int8.
2. **Genome size?** Start small: 1024 genes × 384 dim = 768KB. Fits in L2 cache ENTIRELY.
3. **Metal compute kernels?** For matmul on gene data, use Metal or Accelerate vDSP?
4. **Training?** The genome evolves through: (a) gradient descent on gene data, 
   (b) evolutionary mutation on gene structure, (c) epigenetic updates at runtime.
5. **How to bootstrap?** Train a V5 PID-Net in Python, export to .genome format, 
   then let the C++ organism evolve from there?

---

*The binary is alive. The file is the organism. Memory IS computation.*
*Created: 2026-03-15 by Ion + Harshil*
