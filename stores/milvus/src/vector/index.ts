import { MastraError, ErrorDomain, ErrorCategory } from '@mastra/core/error';
import { createVectorErrorId } from '@mastra/core/storage';
import { MastraVector, validateUpsertInput } from '@mastra/core/vector';
import type {
  QueryResult,
  IndexStats,
  CreateIndexParams,
  UpsertVectorParams,
  QueryVectorParams,
  DescribeIndexParams,
  DeleteIndexParams,
  DeleteVectorParams,
  DeleteVectorsParams,
  UpdateVectorParams,
  SparseVector,
} from '@mastra/core/vector';
import {
  MilvusClient,
  type ClientConfig,
  DataType,
  IndexType,
  RRFRanker,
  type SearchResultData,
  type PartitionData,
} from '@zilliz/milvus2-sdk-node';

import { MilvusFilterTranslator } from './filter';
import type { MilvusVectorFilter } from './filter';

/** Maps Mastra distance metric names to Milvus `metric_type` values for the embedding index. */
const METRIC_MAPPING: Record<string, string> = {
  cosine: 'COSINE',
  euclidean: 'L2',
  dotproduct: 'IP',
};

/** Batch size for SDK `upsert` calls. */
const BATCH_SIZE = 256;

/** Dense float vector field (HNSW). */
const DENSE_VECTOR_FIELD = 'dense_embedding';

/** Sparse field for hybrid search (SPARSE_INVERTED_INDEX / IP). */
const SPARSE_VECTOR_FIELD = 'sparse_embedding';

export type MilvusVectorConfig = ClientConfig & {
  /** The unique identifier for this vector store instance. */
  id: string;
};

/** Parameters for creating a Milvus database (reserved for future use). */
export interface MilvusCreateDatabaseParams {
  /** Database name. */
  name: string;
}

export interface MilvusDeleteVectorParams extends DeleteVectorParams {
  /** Optional Milvus partition name; defaults to `_default` when omitted. */
  partition?: string;
}

export interface MilvusDeleteVectorsParams extends DeleteVectorsParams<MilvusVectorFilter> {
  /** Optional Milvus partition name; applies to ID- and filter-based deletes. */
  partition?: string;
}

export interface MilvusUpsertVectorParams extends UpsertVectorParams {
  /** Optional Milvus partition to write into. */
  partition?: string;
}

/**
 * Query parameters for Milvus.
 * Extends {@link QueryVectorParams} with optional partition scoping for search.
 */
export interface MilvusQueryVectorParams extends QueryVectorParams {
  /** Partition names to search; if omitted, all loaded partitions are considered. */
  partitions?: Array<string>;
}

/**
 * Index statistics for a Milvus collection, including partition metadata from `showPartitions`.
 */
export interface MilvusIndexStats extends IndexStats {
  /** Partition name → partition info as returned by the Milvus SDK. */
  partitions: Record<string, PartitionData>;
}

/**
 * Milvus-specific `updateVector` params: either by primary key or by metadata filter, with optional partition.
 */
type MilvusUpdateVectorParams =
  | {
      indexName: string;
      id: string;
      filter?: never;
      update: { vector?: number[]; metadata?: Record<string, any> };
      partition?: string;
    }
  | {
      indexName: string;
      id?: never;
      filter: MilvusVectorFilter;
      update: { vector?: number[]; metadata?: Record<string, any> };
      partition?: string;
    };

function milvusSparsePayload(sparse?: SparseVector): { indices: number[]; values: number[] } {
  if (!sparse) {
    return { indices: [], values: [] };
  }
  if (sparse.indices.length !== sparse.values.length) {
    throw new Error('sparse vector indices and values must have the same length');
  }
  return { indices: sparse.indices, values: sparse.values };
}

/**
 * Vector store backed by [Milvus](https://milvus.io/).
 *
 * @remarks
 * - Mastra **`indexName`** maps to a Milvus **collection** with fields `id`, `dense_embedding` (FloatVector), `sparse_embedding` (SparseFloatVector), and `metadata` (JSON).
 * - Metrics: `cosine`, `euclidean`, `dotproduct` → HNSW on `embedding`; sparse uses `SPARSE_INVERTED_INDEX` with IP for hybrid.
 * - {@link transformFilter} converts Mastra filters to Milvus boolean expressions on `metadata`.
 * - {@link query} requires `queryVector`; optional `sparseVector` triggers hybrid search (RRF). Metadata-only search is not supported.
 */
export class MilvusVector extends MastraVector<MilvusVectorFilter> {
  private client: MilvusClient;

  /**
   * Creates a new Milvus client wrapper.
   *
   * @param config - Milvus SDK client options plus `id` for this store instance.
   * @see {@link MilvusVectorConfig}
   */
  constructor({ id, ...config }: MilvusVectorConfig) {
    super({ id });
    this.client = new MilvusClient(config);
  }

  /**
   * Loads a collection into query nodes so search and DML can run.
   *
   * @param name - Collection name (Mastra index name).
   * @param action - Logical operation label for error IDs (e.g. `QUERY`, `UPSERT`).
   */
  private async loadCollection(name: string, action: string): Promise<void> {
    try {
      const res = await this.client.loadCollection({ collection_name: name });
      if (res.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', action, 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName: name, reason: res.reason },
        });
      }
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', action, 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName: name },
        },
        error,
      );
    }
  }

  /**
   * Translates a Mastra vector filter into a Milvus boolean expression string, or `undefined` if empty.
   *
   * @param filter - Mongo-style filter; unsupported operators throw during translation when used.
   * @returns Expression for the `filter` argument of search/delete, or `undefined` to omit filtering.
   */
  transformFilter(filter?: MilvusVectorFilter): string | undefined {
    const expr = new MilvusFilterTranslator().translate(filter);
    return expr === '' ? undefined : expr;
  }

  /**
   * Flushes the collection and blocks until Milvus reports affected segments as flushed (SDK `flushSync`).
   *
   * @param indexName - Collection to flush.
   * @throws {MastraError} If the Milvus RPC fails.
   * @remarks Useful after bulk writes when downstream readers rely on persisted segments (e.g. tests).
   */
  async flush({ indexName }: DescribeIndexParams): Promise<void> {
    try {
      await this.client.flushSync({ collection_names: [indexName] });
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'FLUSH', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName },
        },
        error,
      );
    }
  }

  /**
   * Runs approximate nearest-neighbor search on the embedding field.
   *
   * @param params.indexName - Target collection.
   * @param params.queryVector - Query embedding (required).
   * @param params.topK - Number of hits to return (default `10`).
   * @param params.filter - Optional metadata filter (Milvus expression).
   * @param params.includeVector - When true, include stored vectors in results.
   * @param params.partitions - Optional partition subset to search.
   * @param params.sparseVector - When set, runs **hybrid** search (dense + sparse) with RRF reranking.
   * @returns Ranked hits with `id`, `score`, and row payload as `metadata`.
   * @throws {MastraError} If `queryVector` is missing or Milvus returns an error.
   */
  async query({
    indexName,
    queryVector,
    topK = 10,
    filter,
    includeVector = false,
    partitions,
    sparseVector,
  }: MilvusQueryVectorParams): Promise<QueryResult[]> {
    if (!queryVector) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'QUERY', 'MISSING_VECTOR'),
        text: 'queryVector is required for Milvus queries. Metadata-only queries are not supported by this vector store.',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName },
      });
    }
    const translatedFilter = this.transformFilter(filter);
    try {
      await this.loadCollection(indexName, 'QUERY');
      let results: SearchResultData[] = [];
      const output_fields = ['id', 'metadata', DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD];
      if (sparseVector) {
        let sparsePayload: { indices: number[]; values: number[] };
        try {
          sparsePayload = milvusSparsePayload(sparseVector);
        } catch (error) {
          throw new MastraError(
            {
              id: createVectorErrorId('MILVUS', 'QUERY', 'INVALID_SPARSE_VECTOR'),
              domain: ErrorDomain.STORAGE,
              category: ErrorCategory.USER,
              details: { indexName },
            },
            error,
          );
        }
        const res = await this.client.search({
          collection_name: indexName,
          ...(partitions?.length ? { partition_names: partitions } : {}),
          ...(translatedFilter ? { filter: translatedFilter } : {}),
          data: [
            {
              data: queryVector,
              anns_field: DENSE_VECTOR_FIELD,
              topk: topK,
            },
            {
              data: sparsePayload,
              anns_field: SPARSE_VECTOR_FIELD,
              topk: topK,
            },
          ],
          rerank: RRFRanker(),
          topk: topK,
          output_fields,
        });
        results = res.results;
      } else {
        const res = await this.client.search({
          collection_name: indexName,
          ...(partitions?.length ? { partition_names: partitions } : {}),
          ...(translatedFilter ? { filter: translatedFilter } : {}),
          vector: queryVector,
          anns_field: DENSE_VECTOR_FIELD,
          topk: topK,
          output_fields,
        });
        results = res.results;
      }
      return results.map(result => ({
        id: result.id,
        score: result.score,
        metadata: result,
        ...(includeVector && { vector: result[DENSE_VECTOR_FIELD] }),
      }));
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'QUERY', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: {
            indexName,
            topK,
            filter: JSON.stringify(filter),
            hybrid: Boolean(sparseVector),
          },
        },
        error,
      );
    }
  }

  /**
   * Inserts or replaces vectors in batches (SDK `upsert`).
   *
   * @param params.indexName - Target collection.
   * @param params.vectors - Embeddings; dimension must match the collection.
   * @param params.metadata - Optional JSON metadata per row (stored in `metadata` field).
   * @param params.ids - Optional primary keys (VARCHAR); UUIDs generated if omitted.
   * @param params.partition - Optional Milvus partition name.
   * @param params.sparseVectors - Optional sparse rows (same length as `vectors` when provided); omitted rows store an empty sparse vector.
   * @returns The IDs used for all upserted rows.
   * @throws {MastraError} On validation failure or Milvus errors.
   * @remarks Calls {@link flush} after writes so segments (and indexes) see new data reliably.
   */
  async upsert({
    indexName,
    vectors,
    metadata,
    ids,
    partition,
    sparseVectors,
  }: MilvusUpsertVectorParams): Promise<string[]> {
    validateUpsertInput('MILVUS', vectors, metadata, ids);
    if (sparseVectors && sparseVectors.length !== vectors.length) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'UPSERT', 'SPARSE_LENGTH_MISMATCH'),
        text: 'sparseVectors length must match vectors length when provided',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName, vectorCount: vectors.length, sparseCount: sparseVectors.length },
      });
    }

    // Generate IDs if not provided
    const vectorIds = ids || vectors.map(() => crypto.randomUUID());

    const records = vectors.map((vector, i) => {
      let sparsePayload: { indices: number[]; values: number[] };
      try {
        sparsePayload = milvusSparsePayload(sparseVectors?.[i]);
      } catch (error) {
        throw new MastraError(
          {
            id: createVectorErrorId('MILVUS', 'UPSERT', 'INVALID_SPARSE_VECTOR'),
            domain: ErrorDomain.STORAGE,
            category: ErrorCategory.USER,
            details: { indexName, rowIndex: i },
          },
          error,
        );
      }
      return {
        id: vectorIds[i]!,
        [DENSE_VECTOR_FIELD]: vector,
        [SPARSE_VECTOR_FIELD]: sparsePayload,
        metadata: metadata?.[i] ?? {},
      };
    });

    try {
      await this.loadCollection(indexName, 'UPSERT');
      for (let i = 0; i < records.length; i += BATCH_SIZE) {
        const batch = records.slice(i, i + BATCH_SIZE);
        await this.client.upsert({
          collection_name: indexName,
          partition_name: partition,
          data: batch,
        });
      }
      await this.flush({ indexName });
      return vectorIds;
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'UPSERT', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, vectorCount: vectors.length },
        },
        error,
      );
    }
  }

  /**
   * Creates a Milvus collection and HNSW index on `embedding`.
   *
   * @param params.indexName - Collection name.
   * @param params.dimension - Vector dimension (positive integer).
   * @param params.metric - `cosine` | `euclidean` | `dotproduct` (default `cosine`).
   * @throws {MastraError} If arguments are invalid or Milvus returns an error.
   * @remarks Schema: `id` (VARCHAR PK), `dense_embedding` (FloatVector), `sparse_embedding` (SparseFloatVector), `metadata` (JSON).
   */
  async createIndex({ indexName, dimension, metric = 'cosine' }: CreateIndexParams): Promise<void> {
    try {
      if (!Number.isInteger(dimension) || dimension <= 0) {
        throw new Error('Dimension must be a positive integer');
      }
      if (metric && !['cosine', 'euclidean', 'dotproduct'].includes(metric)) {
        throw new Error('Metric must be one of: cosine, euclidean, dotproduct');
      }
    } catch (validationError) {
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'CREATE_INDEX', 'INVALID_ARGS'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.USER,
          details: { indexName, dimension, metric },
        },
        validationError,
      );
    }

    try {
      const collectionRes = await this.client.createCollection({
        collection_name: indexName,
        fields: [
          {
            name: 'id',
            data_type: DataType.VarChar,
            is_primary_key: true,
            max_length: 36,
            autoID: false,
          },
          {
            name: DENSE_VECTOR_FIELD,
            data_type: DataType.FloatVector,
            dim: dimension,
          },
          {
            name: SPARSE_VECTOR_FIELD,
            data_type: DataType.SparseFloatVector,
          },
          {
            name: 'metadata',
            data_type: DataType.JSON,
          },
        ],
        dimension,
      });

      if (collectionRes.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', 'CREATE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, dimension, metric, reason: collectionRes.reason },
        });
      }

      const indexRes = await this.client.createIndex({
        collection_name: indexName,
        field_name: DENSE_VECTOR_FIELD,
        metric_type: METRIC_MAPPING[metric],
        index_type: 'HNSW',
      });

      if (indexRes.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', 'CREATE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, dimension, metric, reason: indexRes.reason },
        });
      }

      const sparseIndexRes = await this.client.createIndex({
        collection_name: indexName,
        field_name: SPARSE_VECTOR_FIELD,
        index_type: IndexType.SPARSE_INVERTED_INDEX,
        metric_type: 'IP',
      });

      if (sparseIndexRes.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', 'CREATE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, dimension, metric, reason: sparseIndexRes.reason },
        });
      }
    } catch (error: any) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'CREATE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, dimension, metric },
        },
        error,
      );
    }
  }

  /**
   * Lists all collection names in the connected Milvus instance (`listCollections`).
   *
   * @returns Collection names usable as Mastra index names.
   * @throws {MastraError} If the Milvus RPC fails.
   */
  async listIndexes(): Promise<string[]> {
    try {
      const indexesResult = await this.client.listCollections();
      return indexesResult?.data.map(({ name }) => name);
    } catch (error) {
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'LIST_INDEXES', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  /**
   * Returns dimension, entity count, and partition details for a collection.
   *
   * @param params.indexName - Collection name.
   * @returns {@link MilvusIndexStats} including `count` from a Milvus `count(*)` query (after load).
   * @throws {MastraError} If the collection is missing or an RPC fails.
   * @remarks Loads the collection so `count(*)` reflects query-visible rows (not only sealed-segment stats).
   */
  async describeIndex({ indexName }: DescribeIndexParams): Promise<MilvusIndexStats> {
    try {
      await this.loadCollection(indexName, 'DESCRIBE_INDEX');

      const count = await this.client.count({ collection_name: indexName });

      const partitions = await this.client.showPartitions({
        collection_name: indexName,
      });

      const description = await this.client.describeCollection({
        collection_name: indexName,
      });

      const vectorField = description.schema.fields.find(f => f.name === DENSE_VECTOR_FIELD);

      if (!vectorField) throw new Error(`No ${DENSE_VECTOR_FIELD} field for collection ${indexName}`);

      return {
        dimension: parseInt(vectorField.dim as string),
        count: count.data,
        partitions: Object.fromEntries(partitions.data.map(partition => [partition.name, partition])),
      };
    } catch (error) {
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'DESCRIBE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName },
        },
        error,
      );
    }
  }

  /**
   * Drops the Milvus collection (`dropCollection`).
   *
   * @param params.indexName - Collection to remove.
   * @throws {MastraError} If Milvus returns an error.
   */
  async deleteIndex({ indexName }: DeleteIndexParams): Promise<void> {
    try {
      const res = await this.client.dropCollection({
        collection_name: indexName,
      });
      if (res.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', 'DELETE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, reason: res.reason },
        });
      }
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'DELETE_INDEX', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName },
        },
        error,
      );
    }
  }

  /**
   * Updates a vector by its ID with the provided vector and/or metadata.
   * @param params - Parameters containing the id for targeting the vector to update
   * @param params.indexName - The name of the index containing the vector.
   * @param params.id - The ID of the vector to update.
   * @param params.update - An object containing the vector and/or metadata to update.
   * @param partition - The partition of the index (optional, Milvus-specific).
   * @returns A promise that resolves when the update is complete.
   * @throws Will throw an error if no updates are provided or if the update operation fails.
   */
  async updateVector({ indexName, update, partition, id, filter }: MilvusUpdateVectorParams): Promise<void> {
    throw new Error('Method not implemented.');
  }

  /**
   * Deletes one vector by primary key.
   *
   * @param params.indexName - Collection name.
   * @param params.id - Primary key to delete.
   * @param params.partition - Optional partition name.
   * @throws {MastraError} If Milvus returns an error.
   */
  async deleteVector({ indexName, id, partition }: MilvusDeleteVectorParams): Promise<void> {
    try {
      await this.loadCollection(indexName, 'DELETE_VECTOR');
      await this.client.delete({
        ids: [id],
        collection_name: indexName,
        partition_name: partition,
      });
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'DELETE_VECTOR', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: {
            indexName,
            id,
          },
        },
        error,
      );
    }
  }

  /**
   * Deletes vectors by explicit IDs or by metadata filter (mutually exclusive).
   *
   * @param params.indexName - Collection name.
   * @param params.ids - Primary keys to delete (non-empty when used).
   * @param params.filter - Mastra filter translated to a Milvus expression for `deleteEntities`.
   * @param params.partition - Optional partition name.
   * @throws {MastraError} If parameters are invalid, both `ids` and `filter` are set, neither is set, or Milvus fails.
   */
  async deleteVectors({ ids, indexName, filter, partition }: MilvusDeleteVectorsParams): Promise<void> {
    // Validate mutually exclusive parameters
    if (ids && filter) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'DELETE_VECTORS', 'MUTUALLY_EXCLUSIVE'),
        text: 'Cannot specify both ids and filter - they are mutually exclusive',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName },
      });
    }

    if (!ids && !filter) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'DELETE_VECTORS', 'NO_TARGET'),
        text: 'Either filter or ids must be provided',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName },
      });
    }

    // Validate ids array is not empty
    if (ids && ids.length === 0) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'DELETE_VECTORS', 'EMPTY_IDS'),
        text: 'Cannot delete with empty ids array',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName },
      });
    }

    // Validate filter is not empty
    if (filter && Object.keys(filter).length === 0) {
      throw new MastraError({
        id: createVectorErrorId('MILVUS', 'DELETE_VECTORS', 'EMPTY_FILTER'),
        text: 'Cannot delete with empty filter object',
        domain: ErrorDomain.STORAGE,
        category: ErrorCategory.USER,
        details: { indexName },
      });
    }
    try {
      await this.loadCollection(indexName, 'DELETE_VECTORS');
      if (ids) {
        await this.client.delete({
          ids: ids,
          collection_name: indexName,
          partition_name: partition,
        });
      } else if (filter) {
        await this.client.deleteEntities({
          collection_name: indexName,
          partition_name: partition,
          filter: this.transformFilter(filter),
        });
      }
    } catch (error) {
      if (error instanceof MastraError) throw error;
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'DELETE_VECTORS', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: {
            indexName,
            ...(filter && { filter: JSON.stringify(filter) }),
            ...(ids && { idsCount: ids.length }),
          },
        },
        error,
      );
    }
  }
}
