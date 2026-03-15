"""
V7 Genome: The Biological Foundation

Binary format for a living genome with:
- Gene types (STRUCTURAL, REGULATORY, RECEPTOR, CHANNEL, SIGNAL, METABOLIC, MODULATORY)
- Promoter regions (TF binding sites)
- UTR regions (splicing signals, mRNA stability)
- Coding regions (the actual weights)
- Epigenetic state (methylation, accessibility)
- Regulatory network (gene-gene interactions)

The genome is mmap'd — the file IS the memory. Zero-copy. Nanosecond access.
Changes persist automatically.
"""

import numpy as np
import mmap
import struct
import os
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# ============================================================
# Constants
# ============================================================

MAGIC = b'GEN7'
VERSION = 7
CODON_SIZE = 16          # float16 values per codon
CODON_BYTES = 32         # 16 × 2 bytes
PROMOTER_CODONS = 2      # 32 float16 values for TF binding sites
UTR_CODONS = 2           # 32 float16 values for splicing/stability signals
MIN_CODING_CODONS = 20   # minimum 320 float16 values
CACHE_LINE = 64          # bytes, for alignment

# Header: 64 bytes (1 cache line)
HEADER_SIZE = 64
# Chromosome entry: 64 bytes (1 cache line)
CHROMOSOME_ENTRY_SIZE = 64
# Gene entry: 128 bytes (2 cache lines — more metadata than V6)
GENE_ENTRY_SIZE = 128


class GeneType(IntEnum):
    STRUCTURAL = 0x00    # → Transform protein (weight matrix)
    REGULATORY = 0x01    # → Transcription factor (controls other genes)
    RECEPTOR   = 0x02    # → Receptor protein (detects signals)
    CHANNEL    = 0x03    # → Ion channel protein (firing control)
    SIGNAL     = 0x04    # → Signaling protein (output encoder)
    METABOLIC  = 0x05    # → Enzyme (state maintenance)
    MODULATORY = 0x06    # → Neuromodulator (brain-wide state)
    SCAFFOLD   = 0x07    # → Structural/connectivity protein


class ChromosomeFunction(IntEnum):
    PERCEPTION  = 0
    MEMORY      = 1
    PREDICTION  = 2
    REGULATION  = 3
    OUTPUT      = 4
    HOMEOSTASIS = 5


# ============================================================
# Binary Structures
# ============================================================

@dataclass
class GenomeHeader:
    """64 bytes, 1 cache line."""
    magic: bytes = MAGIC              # 4 bytes
    version: int = VERSION            # 4 bytes (u32)
    n_chromosomes: int = 0            # 4 bytes (u32)
    n_genes: int = 0                  # 4 bytes (u32)
    d_codon: int = CODON_SIZE         # 4 bytes (u32) — values per codon
    gene_data_offset: int = 0         # 8 bytes (u64) — where gene data starts
    epigenetic_offset: int = 0        # 8 bytes (u64) — where epigenetic state starts
    regulation_offset: int = 0        # 8 bytes (u64) — where regulatory network starts
    total_size: int = 0               # 8 bytes (u64) — total file size
    checksum: int = 0                 # 8 bytes (u64)
    # padding to 64 bytes             # 4 bytes

    STRUCT_FMT = '<4sIIIIQQQQQ4x'
    
    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FMT,
            self.magic, self.version, self.n_chromosomes, self.n_genes,
            self.d_codon, self.gene_data_offset, self.epigenetic_offset,
            self.regulation_offset, self.total_size, self.checksum
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'GenomeHeader':
        vals = struct.unpack(cls.STRUCT_FMT, data[:HEADER_SIZE])
        h = cls()
        h.magic = vals[0]
        h.version = vals[1]
        h.n_chromosomes = vals[2]
        h.n_genes = vals[3]
        h.d_codon = vals[4]
        h.gene_data_offset = vals[5]
        h.epigenetic_offset = vals[6]
        h.regulation_offset = vals[7]
        h.total_size = vals[8]
        h.checksum = vals[9]
        return h


@dataclass
class ChromosomeEntry:
    """64 bytes, 1 cache line."""
    id: int = 0                       # 4 bytes (u32)
    function: int = 0                 # 4 bytes (u32) — ChromosomeFunction
    gene_start: int = 0              # 4 bytes (u32) — first gene index
    gene_count: int = 0              # 4 bytes (u32) — number of genes
    offset: int = 0                  # 8 bytes (u64) — byte offset in file
    size: int = 0                    # 8 bytes (u64) — total data size
    # 32 bytes padding

    STRUCT_FMT = '<IIIIqq32x'
    
    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FMT,
            self.id, self.function, self.gene_start, self.gene_count,
            self.offset, self.size
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'ChromosomeEntry':
        vals = struct.unpack(cls.STRUCT_FMT, data[:CHROMOSOME_ENTRY_SIZE])
        c = cls()
        c.id = vals[0]
        c.function = vals[1]
        c.gene_start = vals[2]
        c.gene_count = vals[3]
        c.offset = vals[4]
        c.size = vals[5]
        return c


@dataclass
class GeneEntry:
    """128 bytes, 2 cache lines. Rich metadata for the Central Dogma."""
    id: int = 0                       # 4 bytes (u32)
    chromosome: int = 0              # 4 bytes (u32)
    gene_type: int = 0              # 4 bytes (u32) — GeneType enum
    
    # Data layout within gene_data section
    data_offset: int = 0             # 8 bytes (u64) — byte offset to this gene's data
    promoter_size: int = 0           # 4 bytes (u32) — number of float16 values
    utr_size: int = 0               # 4 bytes (u32) — number of float16 values  
    coding_size: int = 0            # 4 bytes (u32) — number of float16 values
    total_size: int = 0             # 4 bytes (u32) — total float16 values
    
    # Dimensions (interpretation depends on gene_type)
    d_in: int = 0                    # 4 bytes (u32)
    d_out: int = 0                   # 4 bytes (u32)
    
    # Transcription parameters (stored in gene entry, not in data)
    base_expression: float = 0.0     # 4 bytes (f32) — default expression level
    mRNA_halflife: float = 1.0       # 4 bytes (f32) — mRNA stability (1.0 = stable)
    n_splice_variants: int = 1       # 4 bytes (u32) — how many splice variants
    activity_sensitivity: float = 0.0 # 4 bytes (f32) — how much firing affects expression
    
    # Modulator sensitivity (how neuromodulators affect this gene)
    modulator_sensitivity: float = 0.0  # 4 bytes (f32)
    
    # Flags
    essential: int = 0               # 1 byte (u8) — can this gene be silenced?
    active: int = 1                  # 1 byte (u8) — currently active?
    
    # Padding to 128 bytes
    # 4 + 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1 = 66 bytes
    # padding = 62 bytes
    
    STRUCT_FMT = '<III Q IIII II ffIf f BB 62x'
    
    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FMT,
            self.id, self.chromosome, self.gene_type,
            self.data_offset,
            self.promoter_size, self.utr_size, self.coding_size, self.total_size,
            self.d_in, self.d_out,
            self.base_expression, self.mRNA_halflife, self.n_splice_variants,
            self.activity_sensitivity,
            self.modulator_sensitivity,
            self.essential, self.active
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'GeneEntry':
        vals = struct.unpack(cls.STRUCT_FMT, data[:GENE_ENTRY_SIZE])
        g = cls()
        g.id = vals[0]
        g.chromosome = vals[1]
        g.gene_type = vals[2]
        g.data_offset = vals[3]
        g.promoter_size = vals[4]
        g.utr_size = vals[5]
        g.coding_size = vals[6]
        g.total_size = vals[7]
        g.d_in = vals[8]
        g.d_out = vals[9]
        g.base_expression = vals[10]
        g.mRNA_halflife = vals[11]
        g.n_splice_variants = vals[12]
        g.activity_sensitivity = vals[13]
        g.modulator_sensitivity = vals[14]
        g.essential = vals[15]
        g.active = vals[16]
        return g


# ============================================================
# Genome Class — The Living Binary
# ============================================================

class Genome:
    """
    mmap'd genome binary. The file IS the memory.
    
    Layout:
    ┌─────────────────────┐
    │ Header (64B)        │
    ├─────────────────────┤
    │ Chromosome Table    │
    │ (n_chr × 64B)       │
    ├─────────────────────┤
    │ Gene Table          │
    │ (n_genes × 128B)    │
    ├─────────────────────┤
    │ Gene Data           │
    │ (float16 arrays,    │
    │  cache-line aligned)│
    ├─────────────────────┤
    │ Epigenetic State    │
    │ (float32 per gene)  │
    ├─────────────────────┤
    │ Regulation Network  │
    │ (sparse edges)      │
    └─────────────────────┘
    """
    
    def __init__(self, path: str, mode: str = 'r'):
        """
        Open an existing genome file.
        
        Args:
            path: path to .genome file
            mode: 'r' for read-only, 'rw' for read-write (living genome)
        """
        self.path = path
        self.mode = mode
        
        if mode == 'rw':
            self._fd = open(path, 'r+b')
            self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_WRITE)
        else:
            self._fd = open(path, 'rb')
            self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse header
        self.header = GenomeHeader.unpack(self._mm[:HEADER_SIZE])
        assert self.header.magic == MAGIC, f"Invalid genome file: {self.header.magic}"
        assert self.header.version == VERSION, f"Version mismatch: {self.header.version} != {VERSION}"
        
        # Parse chromosome table
        chr_start = HEADER_SIZE
        self.chromosomes = []
        for i in range(self.header.n_chromosomes):
            offset = chr_start + i * CHROMOSOME_ENTRY_SIZE
            self.chromosomes.append(
                ChromosomeEntry.unpack(self._mm[offset:offset + CHROMOSOME_ENTRY_SIZE])
            )
        
        # Parse gene table
        gene_table_start = chr_start + self.header.n_chromosomes * CHROMOSOME_ENTRY_SIZE
        self.genes = []
        for i in range(self.header.n_genes):
            offset = gene_table_start + i * GENE_ENTRY_SIZE
            self.genes.append(
                GeneEntry.unpack(self._mm[offset:offset + GENE_ENTRY_SIZE])
            )
        
        # Epigenetic state: float32 array (methylation level per gene)
        epi_offset = self.header.epigenetic_offset
        epi_size = self.header.n_genes * 4  # float32
        self._epigenetic_raw = np.frombuffer(
            self._mm, dtype=np.float32, count=self.header.n_genes, offset=epi_offset
        )
        if mode == 'rw':
            # Writable view
            self.epigenetic = np.ndarray(
                self.header.n_genes, dtype=np.float32,
                buffer=self._mm, offset=epi_offset
            )
        else:
            self.epigenetic = self._epigenetic_raw.copy()
    
    def get_gene(self, gene_id: int) -> GeneEntry:
        """Get gene metadata."""
        return self.genes[gene_id]
    
    def get_gene_data(self, gene_id: int) -> np.ndarray:
        """Get raw gene data as float16 array (promoter + UTR + coding)."""
        gene = self.genes[gene_id]
        offset = self.header.gene_data_offset + gene.data_offset
        return np.frombuffer(
            self._mm, dtype=np.float16, count=gene.total_size, offset=offset
        ).copy()  # copy to allow modification without affecting genome
    
    def get_promoter(self, gene_id: int) -> np.ndarray:
        """Get gene's promoter region (TF binding sites)."""
        gene = self.genes[gene_id]
        offset = self.header.gene_data_offset + gene.data_offset
        return np.frombuffer(
            self._mm, dtype=np.float16, count=gene.promoter_size, offset=offset
        ).copy()
    
    def get_utr(self, gene_id: int) -> np.ndarray:
        """Get gene's UTR region (splicing signals)."""
        gene = self.genes[gene_id]
        offset = (self.header.gene_data_offset + gene.data_offset + 
                  gene.promoter_size * 2)  # × 2 for float16 byte size
        return np.frombuffer(
            self._mm, dtype=np.float16, count=gene.utr_size, offset=offset
        ).copy()
    
    def get_coding_region(self, gene_id: int) -> np.ndarray:
        """Get gene's coding region (the actual weights)."""
        gene = self.genes[gene_id]
        offset = (self.header.gene_data_offset + gene.data_offset + 
                  (gene.promoter_size + gene.utr_size) * 2)
        return np.frombuffer(
            self._mm, dtype=np.float16, count=gene.coding_size, offset=offset
        ).copy()
    
    def set_gene_data(self, gene_id: int, data: np.ndarray):
        """Write gene data back to genome (rw mode only)."""
        assert self.mode == 'rw', "Genome not opened in rw mode"
        gene = self.genes[gene_id]
        assert len(data) == gene.total_size, f"Data size mismatch: {len(data)} != {gene.total_size}"
        offset = self.header.gene_data_offset + gene.data_offset
        self._mm[offset:offset + gene.total_size * 2] = data.astype(np.float16).tobytes()
    
    def set_coding_region(self, gene_id: int, data: np.ndarray):
        """Write coding region back to genome (rw mode only)."""
        assert self.mode == 'rw', "Genome not opened in rw mode"
        gene = self.genes[gene_id]
        assert len(data) == gene.coding_size, f"Coding size mismatch: {len(data)} != {gene.coding_size}"
        offset = (self.header.gene_data_offset + gene.data_offset +
                  (gene.promoter_size + gene.utr_size) * 2)
        self._mm[offset:offset + gene.coding_size * 2] = data.astype(np.float16).tobytes()
    
    def get_methylation(self, gene_id: int) -> float:
        """Get epigenetic methylation level (0 = accessible, 1 = silenced)."""
        return float(self.epigenetic[gene_id])
    
    def set_methylation(self, gene_id: int, level: float):
        """Set epigenetic methylation level (rw mode only)."""
        assert self.mode == 'rw', "Genome not opened in rw mode"
        self.epigenetic[gene_id] = np.float32(level)
    
    def get_genes_by_type(self, gene_type: GeneType) -> List[int]:
        """Get all gene IDs of a specific type."""
        return [g.id for g in self.genes if g.gene_type == gene_type]
    
    def get_genes_by_chromosome(self, chr_id: int) -> List[int]:
        """Get all gene IDs in a chromosome."""
        chr_entry = self.chromosomes[chr_id]
        return list(range(chr_entry.gene_start, chr_entry.gene_start + chr_entry.gene_count))
    
    def get_accessible_genes(self, threshold: float = 0.5) -> List[int]:
        """Get all genes whose methylation is below threshold (accessible)."""
        return [i for i in range(self.header.n_genes) if self.epigenetic[i] < threshold]
    
    def get_regulation_network(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get the gene regulatory network.
        Returns: dict mapping gene_id → [(target_gene_id, strength), ...]
        Positive strength = activation, negative = repression.
        """
        reg_offset = self.header.regulation_offset
        # Read number of edges
        n_edges = struct.unpack('<I', self._mm[reg_offset:reg_offset + 4])[0]
        
        network = {}
        offset = reg_offset + 4
        for _ in range(n_edges):
            src, dst = struct.unpack('<II', self._mm[offset:offset + 8])
            weight = struct.unpack('<f', self._mm[offset + 8:offset + 12])[0]
            offset += 12
            
            if src not in network:
                network[src] = []
            network[src].append((dst, weight))
        
        return network
    
    @property
    def n_genes(self) -> int:
        return self.header.n_genes
    
    @property
    def n_chromosomes(self) -> int:
        return self.header.n_chromosomes
    
    def summary(self) -> str:
        """Human-readable genome summary."""
        lines = [
            f"Genome V{self.header.version}: {self.path}",
            f"  Chromosomes: {self.header.n_chromosomes}",
            f"  Genes: {self.header.n_genes}",
            f"  Codon size: {self.header.d_codon}",
            f"  Total size: {self.header.total_size:,} bytes ({self.header.total_size / 1024:.1f} KB)",
            "",
            "  Gene type distribution:"
        ]
        
        type_counts = {}
        for g in self.genes:
            t = GeneType(g.gene_type)
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, count in sorted(type_counts.items(), key=lambda x: x[0]):
            lines.append(f"    {t.name}: {count}")
        
        lines.append("")
        lines.append("  Chromosomes:")
        for c in self.chromosomes:
            func = ChromosomeFunction(c.function)
            lines.append(f"    Chr {c.id} ({func.name}): {c.gene_count} genes")
        
        # Epigenetic state
        accessible = sum(1 for e in self.epigenetic if e < 0.5)
        silenced = self.header.n_genes - accessible
        lines.append(f"\n  Epigenetic state: {accessible} accessible, {silenced} silenced")
        
        return '\n'.join(lines)
    
    def close(self):
        """Close the genome file."""
        # Release numpy references before closing mmap
        self.epigenetic = None
        self._epigenetic_raw = None
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd:
            self._fd.close()
            self._fd = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Genome(genes={self.n_genes}, chromosomes={self.n_chromosomes}, path='{self.path}')"


# ============================================================
# Genome Builder — Creates new genomes
# ============================================================

class GenomeBuilder:
    """
    Builds a new .genome binary file from specification.
    
    Usage:
        builder = GenomeBuilder()
        builder.add_chromosome(ChromosomeFunction.PERCEPTION, [
            GeneSpec(GeneType.STRUCTURAL, d_in=384, d_out=384),
            GeneSpec(GeneType.RECEPTOR, d_in=64, d_out=64),
            GeneSpec(GeneType.CHANNEL, d_in=64, d_out=1),
            GeneSpec(GeneType.REGULATORY, n_targets=4),
        ])
        builder.build("organism.genome")
    """
    
    def __init__(self, d_codon: int = CODON_SIZE):
        self.d_codon = d_codon
        self.chromosomes: List[Tuple[ChromosomeFunction, List['GeneSpec']]] = []
    
    def add_chromosome(self, function: ChromosomeFunction, genes: List['GeneSpec']):
        """Add a chromosome with its genes."""
        self.chromosomes.append((function, genes))
    
    def _align(self, offset: int, alignment: int = CACHE_LINE) -> int:
        """Align offset to cache line boundary."""
        return (offset + alignment - 1) & ~(alignment - 1)
    
    def build(self, path: str, init_method: str = 'xavier') -> Genome:
        """
        Build the genome binary file.
        
        Args:
            path: output file path
            init_method: weight initialization ('xavier', 'normal', 'zeros')
        
        Returns:
            Genome object (opened in rw mode)
        """
        # Count totals
        n_chromosomes = len(self.chromosomes)
        all_genes = []
        for func, genes in self.chromosomes:
            all_genes.extend(genes)
        n_genes = len(all_genes)
        
        # Calculate offsets
        # Header
        offset = HEADER_SIZE
        
        # Chromosome table
        chr_table_offset = offset
        offset += n_chromosomes * CHROMOSOME_ENTRY_SIZE
        
        # Gene table
        gene_table_offset = offset
        offset += n_genes * GENE_ENTRY_SIZE
        
        # Gene data (aligned)
        offset = self._align(offset)
        gene_data_offset = offset
        
        # Calculate per-gene data sizes and offsets
        gene_entries = []
        gene_data_parts = []
        gene_idx = 0
        chr_entries = []
        
        regulation_edges = []  # (src_gene, dst_gene, weight)
        
        for chr_idx, (func, genes) in enumerate(self.chromosomes):
            chr_gene_start = gene_idx
            chr_data_start = offset - gene_data_offset
            
            for spec in genes:
                # Calculate sizes
                promoter_size = PROMOTER_CODONS * self.d_codon  # float16 values
                utr_size = UTR_CODONS * self.d_codon
                coding_size = spec.get_coding_size(self.d_codon)
                total_size = promoter_size + utr_size + coding_size
                
                # Gene data offset (relative to gene_data_offset)
                gene_data_rel = offset - gene_data_offset
                
                # Create gene entry
                entry = GeneEntry(
                    id=gene_idx,
                    chromosome=chr_idx,
                    gene_type=spec.gene_type,
                    data_offset=gene_data_rel,
                    promoter_size=promoter_size,
                    utr_size=utr_size,
                    coding_size=coding_size,
                    total_size=total_size,
                    d_in=spec.d_in,
                    d_out=spec.d_out,
                    base_expression=spec.base_expression,
                    mRNA_halflife=spec.mRNA_halflife,
                    n_splice_variants=spec.n_splice_variants,
                    activity_sensitivity=spec.activity_sensitivity,
                    modulator_sensitivity=spec.modulator_sensitivity,
                    essential=1 if spec.essential else 0,
                    active=1,
                )
                gene_entries.append(entry)
                
                # Generate gene data
                data = self._init_gene_data(spec, promoter_size, utr_size, 
                                           coding_size, init_method)
                gene_data_parts.append(data)
                
                # Track regulatory targets
                if spec.gene_type == GeneType.REGULATORY and spec.regulation_targets:
                    for target, strength in spec.regulation_targets:
                        regulation_edges.append((gene_idx, target, strength))
                
                # Advance offset (aligned)
                offset += total_size * 2  # × 2 for float16 byte size
                offset = self._align(offset)
                gene_idx += 1
            
            chr_entries.append(ChromosomeEntry(
                id=chr_idx,
                function=func,
                gene_start=chr_gene_start,
                gene_count=len(genes),
                offset=gene_data_offset + chr_data_start,
                size=offset - gene_data_offset - chr_data_start,
            ))
        
        # Epigenetic state
        offset = self._align(offset)
        epigenetic_offset = offset
        offset += n_genes * 4  # float32 per gene
        
        # Regulation network
        offset = self._align(offset)
        regulation_offset = offset
        offset += 4  # n_edges (u32)
        offset += len(regulation_edges) * 12  # (u32 src, u32 dst, f32 weight)
        
        total_size = self._align(offset)
        
        # Build header
        header = GenomeHeader(
            n_chromosomes=n_chromosomes,
            n_genes=n_genes,
            d_codon=self.d_codon,
            gene_data_offset=gene_data_offset,
            epigenetic_offset=epigenetic_offset,
            regulation_offset=regulation_offset,
            total_size=total_size,
        )
        
        # Write file
        with open(path, 'wb') as f:
            # Header
            f.write(header.pack())
            
            # Chromosome table
            for c in chr_entries:
                f.write(c.pack())
            
            # Gene table
            for g in gene_entries:
                f.write(g.pack())
            
            # Pad to gene data offset
            pos = f.tell()
            if pos < gene_data_offset:
                f.write(b'\x00' * (gene_data_offset - pos))
            
            # Gene data
            for i, data in enumerate(gene_data_parts):
                pos = f.tell()
                expected = gene_data_offset + gene_entries[i].data_offset
                if pos < expected:
                    f.write(b'\x00' * (expected - pos))
                f.write(data.astype(np.float16).tobytes())
            
            # Pad to epigenetic offset
            pos = f.tell()
            if pos < epigenetic_offset:
                f.write(b'\x00' * (epigenetic_offset - pos))
            
            # Epigenetic state (all accessible = 0.0)
            epi = np.zeros(n_genes, dtype=np.float32)
            f.write(epi.tobytes())
            
            # Pad to regulation offset
            pos = f.tell()
            if pos < regulation_offset:
                f.write(b'\x00' * (regulation_offset - pos))
            
            # Regulation network
            f.write(struct.pack('<I', len(regulation_edges)))
            for src, dst, weight in regulation_edges:
                f.write(struct.pack('<IIf', src, dst, weight))
            
            # Pad to total size
            pos = f.tell()
            if pos < total_size:
                f.write(b'\x00' * (total_size - pos))
        
        # Open and return
        return Genome(path, mode='rw')
    
    def _init_gene_data(self, spec: 'GeneSpec', promoter_size: int, 
                        utr_size: int, coding_size: int, method: str) -> np.ndarray:
        """Initialize gene data (promoter + UTR + coding)."""
        total = promoter_size + utr_size + coding_size
        
        # Promoter: random patterns for TF binding sites
        # Each gene gets a unique "signature" in its promoter
        promoter = np.random.randn(promoter_size).astype(np.float32)
        promoter /= np.linalg.norm(promoter) + 1e-8  # normalize
        
        # UTR: splicing signals (initialized near 1.0 = include all codons)
        utr = np.ones(utr_size, dtype=np.float32) * 0.5
        utr += np.random.randn(utr_size).astype(np.float32) * 0.1
        
        # Coding region: depends on gene type
        if method == 'xavier' and spec.d_in > 0 and spec.d_out > 0:
            # Xavier/Glorot initialization
            std = np.sqrt(2.0 / (spec.d_in + spec.d_out))
            coding = np.random.randn(coding_size).astype(np.float32) * std
        elif method == 'normal':
            coding = np.random.randn(coding_size).astype(np.float32) * 0.02
        else:
            coding = np.zeros(coding_size, dtype=np.float32)
        
        data = np.concatenate([promoter, utr, coding])
        return data.astype(np.float16)


# ============================================================
# Gene Specification (for building genomes)
# ============================================================

@dataclass
class GeneSpec:
    """Specification for a single gene."""
    gene_type: GeneType
    d_in: int = 0
    d_out: int = 0
    n_coding_codons: int = 0       # override coding size (0 = auto-calculate)
    base_expression: float = 0.5
    mRNA_halflife: float = 0.95
    n_splice_variants: int = 1
    activity_sensitivity: float = 0.1
    modulator_sensitivity: float = 0.1
    essential: bool = False
    regulation_targets: Optional[List[Tuple[int, float]]] = None  # (target_gene_id, strength)
    
    def get_coding_size(self, d_codon: int) -> int:
        """Calculate coding region size in float16 values."""
        if self.n_coding_codons > 0:
            return self.n_coding_codons * d_codon
        
        # Auto-calculate based on gene type
        if self.gene_type == GeneType.STRUCTURAL:
            # Weight matrix: d_in × d_out + bias
            return self.d_in * self.d_out + self.d_out
        
        elif self.gene_type == GeneType.REGULATORY:
            # Binding pattern (128) + target info (96) + strength (64) = 288
            return 288
        
        elif self.gene_type == GeneType.RECEPTOR:
            # Selectivity (64) + sensitivity (1) + response_type (1) = 66
            return max(66, self.d_in)
        
        elif self.gene_type == GeneType.CHANNEL:
            # Threshold (1) + conductance (1) + refractory (1) + gate_weights (64) = 67
            return max(67, self.d_in)
        
        elif self.gene_type == GeneType.SIGNAL:
            # Encoding matrix: d_in × d_out + signal_type (1)
            return self.d_in * self.d_out + 1
        
        elif self.gene_type == GeneType.METABOLIC:
            # Transform matrix: d × d + target_state (d) 
            return self.d_in * self.d_in + self.d_in
        
        elif self.gene_type == GeneType.MODULATORY:
            # Pattern (64) + diffusion_range (1) + effects (65) = 130
            return 130
        
        elif self.gene_type == GeneType.SCAFFOLD:
            # Connectivity weights: d × d
            return self.d_in * self.d_out if self.d_in > 0 else 256
        
        # Fallback
        return MIN_CODING_CODONS * d_codon


# ============================================================
# Preset Genome Blueprints
# ============================================================

def create_minimal_genome(d_model: int = 64, path: str = "minimal.genome") -> Genome:
    """
    Minimal viable genome for testing.
    6 chromosomes, ~60 genes, ~100KB.
    Small d_model for fast iteration.
    """
    builder = GenomeBuilder()
    
    # Chromosome 0: PERCEPTION
    builder.add_chromosome(ChromosomeFunction.PERCEPTION, [
        # Structural genes for input processing
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.8, essential=True),
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.6),
        # Receptor genes for sensory input
        GeneSpec(GeneType.RECEPTOR, d_in=d_model, base_expression=0.9, essential=True),
        GeneSpec(GeneType.RECEPTOR, d_in=d_model, base_expression=0.7),
        # Channel genes for firing control
        GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.8),
        # Regulatory gene
        GeneSpec(GeneType.REGULATORY, base_expression=0.5),
    ])
    
    # Chromosome 1: MEMORY
    builder.add_chromosome(ChromosomeFunction.MEMORY, [
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.7),
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.5),
        GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.6),
        GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.7),
        GeneSpec(GeneType.REGULATORY, base_expression=0.5),
    ])
    
    # Chromosome 2: PREDICTION
    builder.add_chromosome(ChromosomeFunction.PREDICTION, [
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.7),
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.5),
        GeneSpec(GeneType.SIGNAL, d_in=d_model, d_out=d_model, base_expression=0.6),
        GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.6),
        GeneSpec(GeneType.REGULATORY, base_expression=0.5),
    ])
    
    # Chromosome 3: REGULATION
    builder.add_chromosome(ChromosomeFunction.REGULATION, [
        GeneSpec(GeneType.REGULATORY, base_expression=0.8, essential=True),
        GeneSpec(GeneType.REGULATORY, base_expression=0.7),
        GeneSpec(GeneType.REGULATORY, base_expression=0.6),
        GeneSpec(GeneType.MODULATORY, base_expression=0.5),
        GeneSpec(GeneType.MODULATORY, base_expression=0.4),
    ])
    
    # Chromosome 4: OUTPUT
    builder.add_chromosome(ChromosomeFunction.OUTPUT, [
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.8, essential=True),
        GeneSpec(GeneType.STRUCTURAL, d_in=d_model, d_out=d_model, base_expression=0.6),
        GeneSpec(GeneType.SIGNAL, d_in=d_model, d_out=d_model, base_expression=0.7, essential=True),
        GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.7),
        GeneSpec(GeneType.REGULATORY, base_expression=0.5),
    ])
    
    # Chromosome 5: HOMEOSTASIS
    builder.add_chromosome(ChromosomeFunction.HOMEOSTASIS, [
        GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.7, essential=True),
        GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.5),
        GeneSpec(GeneType.REGULATORY, base_expression=0.6),
        GeneSpec(GeneType.MODULATORY, base_expression=0.4),
    ])
    
    return builder.build(path)


def create_standard_genome(d_model: int = 128, path: str = "standard.genome") -> Genome:
    """
    Standard genome for language modeling.
    6 chromosomes, ~120 genes, ~2MB.
    """
    builder = GenomeBuilder()
    n_structural = 8  # structural genes per functional chromosome
    n_regulatory = 4
    
    for func in ChromosomeFunction:
        genes = []
        
        # Structural genes (the workhorses)
        for i in range(n_structural):
            genes.append(GeneSpec(
                GeneType.STRUCTURAL, 
                d_in=d_model, d_out=d_model,
                base_expression=0.8 - i * 0.05,
                essential=(i < 2),
            ))
        
        # Regulatory genes
        for i in range(n_regulatory):
            genes.append(GeneSpec(
                GeneType.REGULATORY,
                base_expression=0.6 - i * 0.1,
            ))
        
        # Type-specific genes
        if func == ChromosomeFunction.PERCEPTION:
            genes.extend([
                GeneSpec(GeneType.RECEPTOR, d_in=d_model, base_expression=0.9, essential=True),
                GeneSpec(GeneType.RECEPTOR, d_in=d_model, base_expression=0.7),
                GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.8),
            ])
        elif func == ChromosomeFunction.MEMORY:
            genes.extend([
                GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.7),
                GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.5),
            ])
        elif func == ChromosomeFunction.PREDICTION:
            genes.extend([
                GeneSpec(GeneType.SIGNAL, d_in=d_model, d_out=d_model, base_expression=0.7),
            ])
        elif func == ChromosomeFunction.REGULATION:
            genes.extend([
                GeneSpec(GeneType.MODULATORY, base_expression=0.6),
                GeneSpec(GeneType.MODULATORY, base_expression=0.4),
            ])
        elif func == ChromosomeFunction.OUTPUT:
            genes.extend([
                GeneSpec(GeneType.SIGNAL, d_in=d_model, d_out=d_model, base_expression=0.8, essential=True),
                GeneSpec(GeneType.CHANNEL, d_in=d_model, base_expression=0.7),
            ])
        elif func == ChromosomeFunction.HOMEOSTASIS:
            genes.extend([
                GeneSpec(GeneType.METABOLIC, d_in=d_model, base_expression=0.8, essential=True),
            ])
        
        builder.add_chromosome(func, genes)
    
    return builder.build(path)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import tempfile
    import time
    
    print("=" * 70)
    print("V7 Genome — The Biological Foundation")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Create minimal genome
        print("\n--- Creating Minimal Genome (d=64) ---")
        t0 = time.perf_counter()
        path = os.path.join(tmpdir, "minimal.genome")
        genome = create_minimal_genome(d_model=64, path=path)
        t1 = time.perf_counter()
        print(f"  Created in {(t1-t0)*1000:.1f}ms")
        print(genome.summary())
        
        # Test 2: Gene access
        print("\n--- Gene Access ---")
        for gene_id in range(min(5, genome.n_genes)):
            gene = genome.get_gene(gene_id)
            t0 = time.perf_counter()
            data = genome.get_gene_data(gene_id)
            t1 = time.perf_counter()
            promoter = genome.get_promoter(gene_id)
            coding = genome.get_coding_region(gene_id)
            
            print(f"  Gene {gene_id} ({GeneType(gene.gene_type).name}): "
                  f"promoter={len(promoter)}, coding={len(coding)}, "
                  f"total={len(data)} float16 values, "
                  f"access={((t1-t0)*1e9):.0f}ns")
        
        # Test 3: Epigenetic state
        print("\n--- Epigenetic State ---")
        print(f"  Initial methylation: {[f'{genome.get_methylation(i):.2f}' for i in range(5)]}")
        genome.set_methylation(2, 0.8)  # silence gene 2
        genome.set_methylation(3, 0.95)  # deeply silence gene 3
        print(f"  After modification: {[f'{genome.get_methylation(i):.2f}' for i in range(5)]}")
        accessible = genome.get_accessible_genes()
        print(f"  Accessible genes: {len(accessible)}/{genome.n_genes}")
        
        # Test 4: Gene type queries
        print("\n--- Gene Type Queries ---")
        for gt in GeneType:
            ids = genome.get_genes_by_type(gt)
            if ids:
                print(f"  {gt.name}: {len(ids)} genes (IDs: {ids[:5]}{'...' if len(ids) > 5 else ''})")
        
        # Test 5: Chromosome queries
        print("\n--- Chromosome Queries ---")
        for chr_id in range(genome.n_chromosomes):
            chr_entry = genome.chromosomes[chr_id]
            gene_ids = genome.get_genes_by_chromosome(chr_id)
            func = ChromosomeFunction(chr_entry.function)
            print(f"  Chr {chr_id} ({func.name}): genes {gene_ids}")
        
        # Test 6: Regulatory network
        print("\n--- Regulatory Network ---")
        reg_net = genome.get_regulation_network()
        if reg_net:
            for src, targets in reg_net.items():
                for dst, w in targets:
                    print(f"  Gene {src} → Gene {dst} (weight={w:.3f})")
        else:
            print("  No regulatory edges (will be populated during training)")
        
        # Test 7: Write and verify
        print("\n--- Persistence Test ---")
        genome.set_methylation(0, 0.42)
        original_meth = genome.get_methylation(0)
        genome.close()
        
        # Reopen and verify
        genome2 = Genome(path, mode='r')
        restored_meth = genome2.get_methylation(0)
        assert abs(original_meth - restored_meth) < 1e-6, "Persistence failed!"
        print(f"  Methylation persisted: {restored_meth:.4f} ✓")
        genome2.close()
        
        # Test 8: Standard genome
        print("\n--- Creating Standard Genome (d=128) ---")
        t0 = time.perf_counter()
        path2 = os.path.join(tmpdir, "standard.genome")
        genome3 = create_standard_genome(d_model=128, path=path2)
        t1 = time.perf_counter()
        print(f"  Created in {(t1-t0)*1000:.1f}ms")
        print(genome3.summary())
        genome3.close()
        
        # Test 9: Bulk access benchmark
        print("\n--- Bulk Access Benchmark ---")
        genome4 = Genome(path2)
        t0 = time.perf_counter()
        n_accesses = 10000
        for _ in range(n_accesses):
            gene_id = np.random.randint(0, genome4.n_genes)
            _ = genome4.get_coding_region(gene_id)
        t1 = time.perf_counter()
        total_us = (t1 - t0) * 1e6
        print(f"  {n_accesses} random gene accesses in {total_us:.0f}μs ({total_us/n_accesses:.1f}μs per access)")
        print(f"  = {n_accesses / (t1-t0):.0f} gene accesses/second")
        genome4.close()
    
    print("\n✅ V7 Genome infrastructure complete!")
