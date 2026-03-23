import { MastraError } from '@mastra/core/error';
import { createVector, VECTOR_DIMENSION } from '../../../_test-utils/src/domains/vector/test-helpers';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { MilvusVector } from './index';

const milvusAddress = process.env.MILVUS_ADDRESS || '127.0.0.1:19530';

describe('MilvusVector', () => {
  let milvus: MilvusVector;

  beforeAll(() => {
    milvus = new MilvusVector({
      id: 'milvus-test',
      address: milvusAddress,
    });
  });

  it('should construct with id and address', () => {
    expect(() => new MilvusVector({ id: 'milvus-ctor', address: milvusAddress })).not.toThrow();
  });

  describe('createIndex validation', () => {
    it('should reject non-positive dimension', async () => {
      await expect(
        milvus.createIndex({
          indexName: `invalid_dim_${Date.now()}`,
          dimension: 0,
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject invalid metric', async () => {
      await expect(
        milvus.createIndex({
          indexName: `invalid_metric_${Date.now()}`,
          dimension: 4,
          metric: 'l2' as 'cosine',
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject non-integer dimension', async () => {
      await expect(
        milvus.createIndex({
          indexName: `invalid_dim_float_${Date.now()}`,
          dimension: 3.5,
        }),
      ).rejects.toThrow(MastraError);
    });
  });

  describe('upsert validation', () => {
    it('should reject empty vectors array', async () => {
      await expect(
        milvus.upsert({
          indexName: 'any',
          vectors: [],
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject metadata length mismatch', async () => {
      await expect(
        milvus.upsert({
          indexName: 'any',
          vectors: [createVector(1), createVector(2)],
          metadata: [{ a: 1 }],
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject ids length mismatch', async () => {
      await expect(
        milvus.upsert({
          indexName: 'any',
          vectors: [createVector(1)],
          metadata: [{ a: 1 }],
          ids: ['a', 'b'],
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject sparseVectors length mismatch', async () => {
      await expect(
        milvus.upsert({
          indexName: 'any',
          vectors: [createVector(1), createVector(2)],
          sparseVectors: [{ indices: [1], values: [1] }],
        }),
      ).rejects.toThrow(MastraError);
    });

    it('should reject sparse row with indices/values length mismatch on upsert', async () => {
      await expect(
        milvus.upsert({
          indexName: 'any',
          vectors: [createVector(1)],
          sparseVectors: [{ indices: [1, 2], values: [1] }],
        }),
      ).rejects.toThrow(MastraError);
    });
  });

  describe('Integration', () => {
    const uniqueIndexName = () => `mastra_milvus_test_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    const flushIndex = (indexName: string) => milvus.flush({ indexName });

    let indexToCleanup: string | undefined;

    afterEach(async () => {
      if (!indexToCleanup) {
        return;
      }
      try {
        await milvus.deleteIndex({ indexName: indexToCleanup });
      } catch {
        // collection may already be dropped by the test
      }
      indexToCleanup = undefined;
    });

    it('should create an index and include it in listIndexes', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
        metric: 'cosine',
      });
      await flushIndex(indexToCleanup!);

      const listed = await milvus.listIndexes();
      expect(Array.isArray(listed)).toBe(true);
      expect((listed ?? []).every(n => typeof n === 'string')).toBe(true);
      expect(listed).toContain(indexToCleanup);
    }, 120000);

    it('should describe a new index with expected dimension, zero rows, and default partition', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const stats = await milvus.describeIndex({ indexName: indexToCleanup });
      expect(stats.dimension).toBe(VECTOR_DIMENSION);
      expect(stats.count).toBe(0);
      expect(stats.partitions).toHaveProperty('_default');
    }, 120000);

    it('should upsert vectors with metadata and update row count', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const ids = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(1), createVector(2)],
        metadata: [{ tag: 'a' }, { tag: 'b' }],
      });
      expect(ids).toHaveLength(2);
      ids.forEach(id => expect(typeof id).toBe('string'));

      await flushIndex(indexToCleanup!);

      const stats = await milvus.describeIndex({ indexName: indexToCleanup });
      expect(stats.count).toBe(2);
    }, 120000);

    it('should upsert vectors without metadata', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const ids = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(11), createVector(12)],
      });
      expect(ids).toHaveLength(2);

      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(11),
        topK: 2,
      });
      expect(results.length).toBeGreaterThan(0);
      expect(results.some(r => r.id === ids[0])).toBe(true);
    }, 120000);

    it('should query with topK and return scores in descending order', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(20), createVector(21), createVector(22)],
        metadata: [{ n: 0 }, { n: 1 }, { n: 2 }],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(21),
        topK: 3,
      });
      expect(results.length).toBeGreaterThan(0);
      expect(results.length).toBeLessThanOrEqual(3);
      for (let i = 0; i < results.length - 1; i++) {
        expect(results[i]!.score).toBeGreaterThanOrEqual(results[i + 1]!.score ?? 0);
      }
    }, 120000);

    it('should return embeddings when includeVector is true', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(30)],
        metadata: [{ only: true }],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(30),
        topK: 1,
        includeVector: true,
      });
      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty('vector');
      expect(Array.isArray(results[0]!.vector)).toBe(true);
      expect(results[0]!.vector!.length).toBe(VECTOR_DIMENSION);
    }, 120000);

    it('should delete a vector by id and decrease row count', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [id0, id1] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(40), createVector(41)],
        metadata: [{ i: 0 }, { i: 1 }],
      });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(2);

      await milvus.deleteVector({ indexName: indexToCleanup, id: id0! });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(1);

      const remaining = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(41),
        topK: 5,
      });
      expect(remaining.map(r => r.id)).toContain(id1);
      expect(remaining.map(r => r.id)).not.toContain(id0);
    }, 120000);

    it('should delete multiple vectors by ids', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const ids = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(50), createVector(51), createVector(52)],
      });
      await flushIndex(indexToCleanup!);
      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(3);

      await milvus.deleteVectors({ indexName: indexToCleanup, ids: [ids[0]!, ids[1]!] });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(1);
    }, 120000);

    it('should create an index with euclidean metric and run query', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
        metric: 'euclidean',
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({ indexName: indexToCleanup, vectors: [createVector(60)] });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(60),
        topK: 1,
      });
      expect(results).toHaveLength(1);
    }, 120000);

    it('should create an index with dotproduct metric and run query', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
        metric: 'dotproduct',
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({ indexName: indexToCleanup, vectors: [createVector(61)] });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(61),
        topK: 1,
      });
      expect(results).toHaveLength(1);
    }, 120000);

    it('should upsert more than one batch when vector count exceeds batch size', async () => {
      indexToCleanup = uniqueIndexName();
      const batchCount = 260;

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const vectors = Array.from({ length: batchCount }, (_, i) => createVector(i % 40));
      await milvus.upsert({ indexName: indexToCleanup, vectors });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(batchCount);
    }, 180000);

    it('should pass partition_names when querying with partitions option', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(80)],
        metadata: [{ p: true }],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(80),
        topK: 1,
        partitions: ['_default'],
      });
      expect(results).toHaveLength(1);
    }, 120000);

    it('should hybrid search when sparseVector is set (dense + sparse RRF)', async () => {
      indexToCleanup = uniqueIndexName();
      const sparseDim = 100;
      const sparseVec = { indices: [sparseDim], values: [1.0] };

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(90)],
        sparseVectors: [sparseVec],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(90),
        topK: 5,
        sparseVector: sparseVec,
      });
      expect(results.length).toBeGreaterThanOrEqual(1);
      expect(results[0]?.id).toBeDefined();
    }, 120000);

    it('should upsert mixed sparse rows (empty + non-empty) and hybrid-query the non-empty row', async () => {
      indexToCleanup = uniqueIndexName();
      const sparseDim = 210;
      const sparseVec = { indices: [sparseDim], values: [2.0] };

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [idEmpty, idSparse] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(91), createVector(92)],
        metadata: [{ slot: 'no_sparse' }, { slot: 'has_sparse' }],
        sparseVectors: [{ indices: [], values: [] }, sparseVec],
      });
      await flushIndex(indexToCleanup!);

      const hybrid = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(92),
        topK: 4,
        sparseVector: sparseVec,
      });
      expect(hybrid.map(r => r.id)).toContain(idSparse);
      expect(hybrid.map(r => r.id)).toContain(idEmpty);

      const denseOnly = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(91),
        topK: 2,
      });
      expect(denseOnly.map(r => r.id)).toContain(idEmpty);
    }, 120000);

    it('should include dense vector in hybrid query when includeVector is true', async () => {
      indexToCleanup = uniqueIndexName();
      const sparseDim = 220;
      const sparseVec = { indices: [sparseDim], values: [1.0] };
      const dense = createVector(93);

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [dense],
        sparseVectors: [sparseVec],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: dense,
        topK: 2,
        sparseVector: sparseVec,
        includeVector: true,
      });
      expect(results[0]?.vector).toBeDefined();
      expect(results[0]!.vector!.length).toBe(VECTOR_DIMENSION);
    }, 120000);

    it('should hybrid search with metadata filter', async () => {
      indexToCleanup = uniqueIndexName();
      const dA = 230;
      const dB = 231;
      const sparseA = { indices: [dA], values: [1.0] };
      const sparseB = { indices: [dB], values: [1.0] };

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(94), createVector(95)],
        metadata: [{ branch: 'a' }, { branch: 'b' }],
        sparseVectors: [sparseA, sparseB],
      });
      await flushIndex(indexToCleanup!);

      const filtered = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(95),
        topK: 5,
        sparseVector: sparseB,
        filter: { branch: 'a' },
      });
      expect(filtered).toHaveLength(1);
      const meta = filtered[0]!.metadata as Record<string, unknown> | undefined;
      expect(meta?.metadata).toEqual({ branch: 'a' });
    }, 120000);

    it('should hybrid search with partitions option', async () => {
      indexToCleanup = uniqueIndexName();
      const sparseDim = 240;
      const sparseVec = { indices: [sparseDim], values: [1.0] };

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(96)],
        metadata: [{ p: true }],
        partition: '_default',
        sparseVectors: [sparseVec],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(96),
        topK: 2,
        sparseVector: sparseVec,
        partitions: ['_default'],
      });
      expect(results.length).toBeGreaterThanOrEqual(1);
    }, 120000);

    it('should surface sparse channel in hybrid results when query sparse matches a different row than dense', async () => {
      indexToCleanup = uniqueIndexName();
      const dimOnlyB = 250;
      const sparseForB = { indices: [dimOnlyB], values: [1.0] };
      const sparseForA = { indices: [251], values: [1.0] };

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [idA, idB] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(97), createVector(98)],
        sparseVectors: [sparseForA, sparseForB],
      });
      await flushIndex(indexToCleanup!);

      const results = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(97),
        topK: 4,
        sparseVector: sparseForB,
      });
      expect(results.map(r => r.id)).toContain(idA);
      expect(results.map(r => r.id)).toContain(idB);
    }, 120000);

    it('should upsert with partition option targeting default partition', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(100)],
        metadata: [{ part: '_default' }],
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      const hits = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(100),
        topK: 1,
        partitions: ['_default'],
      });
      expect(hits).toHaveLength(1);
    }, 120000);

    it('should update vector metadata by id', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [id] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(110)],
        metadata: [{ version: 1, label: 'orig' }],
      });
      await flushIndex(indexToCleanup!);

      await milvus.updateVector({
        indexName: indexToCleanup,
        id: id!,
        update: { metadata: { version: 2, label: 'updated' } },
      });
      await flushIndex(indexToCleanup!);

      const rows = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(110),
        topK: 1,
      });
      expect(rows[0]?.id).toBe(id);
      const row = rows[0]?.metadata as Record<string, unknown> | undefined;
      expect(row?.metadata).toEqual({ version: 2, label: 'updated' });
    }, 120000);

    it('should update vector metadata by id with partition option', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [id] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(111)],
        metadata: [{ slot: 'p' }],
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      await milvus.updateVector({
        indexName: indexToCleanup,
        id: id!,
        partition: '_default',
        update: { metadata: { slot: 'q' } },
      });
      await flushIndex(indexToCleanup!);

      const rows = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(111),
        topK: 1,
        partitions: ['_default'],
      });
      const row = rows[0]?.metadata as Record<string, unknown> | undefined;
      expect(row?.metadata).toEqual({ slot: 'q' });
    }, 120000);

    it('should update vector embedding by id', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [id] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(120)],
        metadata: [{ n: 0 }],
      });
      await flushIndex(indexToCleanup!);

      const newVec = createVector(121);
      await milvus.updateVector({
        indexName: indexToCleanup,
        id: id!,
        update: { vector: newVec },
      });
      await flushIndex(indexToCleanup!);

      const rows = await milvus.query({
        indexName: indexToCleanup,
        queryVector: newVec,
        topK: 1,
        includeVector: true,
      });
      expect(rows[0]?.id).toBe(id);
    }, 120000);

    it('should resolve updateVector with non-empty filter', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(130)],
        metadata: [{ tag: 'keep' }],
      });
      await flushIndex(indexToCleanup!);

      await expect(
        milvus.updateVector({
          indexName: indexToCleanup,
          filter: { tag: 'keep' },
          update: { metadata: { tag: 'x' } },
        }),
      ).resolves.toBeUndefined();
    }, 120000);

    it('should round-trip explicit ids on upsert and query', async () => {
      indexToCleanup = uniqueIndexName();
      const explicitId = '00000000-0000-4000-8000-000000000099';

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const ids = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(70)],
        metadata: [{ fixed: true }],
        ids: [explicitId],
      });
      expect(ids).toEqual([explicitId]);

      await flushIndex(indexToCleanup!);

      const hits = await milvus.query({
        indexName: indexToCleanup,
        queryVector: createVector(70),
        topK: 1,
      });
      expect(hits[0]?.id).toBe(explicitId);
    }, 120000);

    it('should remove an index from listIndexes after deleteIndex', async () => {
      const indexName = uniqueIndexName();

      await milvus.createIndex({
        indexName,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexName);
      expect(await milvus.listIndexes()).toContain(indexName);

      await milvus.deleteIndex({ indexName });

      const listed = await milvus.listIndexes();
      expect(listed).not.toContain(indexName);
    }, 120000);

    it('should throw when query is missing queryVector', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await expect(
        milvus.query({
          indexName: indexToCleanup,
          queryVector: undefined as unknown as number[],
          topK: 1,
        }),
      ).rejects.toThrow(MastraError);
    }, 120000);

    it('should throw when hybrid query sparseVector has indices/values length mismatch', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await expect(
        milvus.query({
          indexName: indexToCleanup,
          queryVector: createVector(1),
          topK: 2,
          sparseVector: { indices: [1, 2], values: [1] },
        }),
      ).rejects.toThrow(MastraError);
    }, 120000);

    it('should throw when updateVector targets a missing id', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      await expect(
        milvus.updateVector({
          indexName: indexToCleanup,
          id: '00000000-0000-4000-8000-00000000dead',
          update: { metadata: { x: 1 } },
        }),
      ).rejects.toThrow(MastraError);
    }, 120000);

    it('should throw describeIndex for a non-existent collection', async () => {
      await expect(milvus.describeIndex({ indexName: `no_such_collection_${Date.now()}` })).rejects.toThrow(
        MastraError,
      );
    }, 120000);

    it('should throw when querying a non-existent collection', async () => {
      await expect(
        milvus.query({
          indexName: `missing_collection_${Date.now()}`,
          queryVector: createVector(1),
          topK: 1,
        }),
      ).rejects.toThrow(MastraError);
    }, 120000);

    it('should delete a vector with partition option', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const [id] = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(140)],
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(1);

      await milvus.deleteVector({
        indexName: indexToCleanup,
        id: id!,
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(0);
    }, 120000);

    it('should delete vectors by ids with partition option', async () => {
      indexToCleanup = uniqueIndexName();

      await milvus.createIndex({
        indexName: indexToCleanup,
        dimension: VECTOR_DIMENSION,
      });
      await flushIndex(indexToCleanup!);

      const ids = await milvus.upsert({
        indexName: indexToCleanup,
        vectors: [createVector(150), createVector(151)],
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      await milvus.deleteVectors({
        indexName: indexToCleanup,
        ids: [ids[0]!],
        partition: '_default',
      });
      await flushIndex(indexToCleanup!);

      expect((await milvus.describeIndex({ indexName: indexToCleanup })).count).toBe(1);
    }, 120000);
  });

  describe('updateVector validation', () => {
    const placeholderIndexName = 'mastra_milvus_update_placeholder';

    it('should reject empty update payload', async () => {
      await expect(
        milvus.updateVector({
          indexName: placeholderIndexName,
          id: 'any',
          update: {},
        }),
      ).rejects.toThrow(/No updates provided/i);
    });
  });

  describe('deleteVectors validation', () => {
    it('should reject empty ids array', async () => {
      await expect(
        milvus.deleteVectors({
          indexName: 'any',
          ids: [],
        }),
      ).rejects.toThrow(/empty ids/i);
    });
  });
});
