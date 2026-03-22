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
} from '@mastra/core/vector';
import {
  MilvusClient,
  type ClientConfig,
  DataType,
  type SearchResultData,
  type PartitionData,
} from '@zilliz/milvus2-sdk-node';

import { MilvusFilterTranslator } from './filter';
import type { MilvusVectorFilter } from './filter';

const METRIC_MAPPING: Record<string, string> = {
  cosine: 'COSINE',
  euclidean: 'L2',
  dotproduct: 'IP',
};

const BATCH_SIZE = 256;

export type MilvusVectorConfig = ClientConfig & {
  /** The unique identifier for this vector store instance. */
  id: string;
};

export interface MilvusCreateDatabaseParams {
  name: string;
}

export interface MilvusCreateIndexParams extends CreateIndexParams {
  maxLength: number;
}

export interface MilvusDeleteVectorParams extends DeleteVectorParams {
  partition?: string;
}

export interface MilvusDeleteVectorsParams extends DeleteVectorsParams<MilvusVectorFilter> {
  partition?: string;
}

export interface MilvusUpsertVectorParams extends UpsertVectorParams {
  partition?: string;
}

export interface MilvusQueryVectorParams extends QueryVectorParams {
  partitions?: Array<string>;
}

export interface MilvusIndexStats extends IndexStats {
  partitions: Record<string, PartitionData>;
}

export class MilvusVector extends MastraVector<MilvusVectorFilter> {
  private client: MilvusClient;

  /**
   * Creates a new MilvusVector client.
   *
   * @param config - Configuration options for the Milvus client.
   * @see {@link MilvusVectorConfig} for all available options.
   */
  constructor({ id, ...config }: MilvusVectorConfig) {
    super({ id });
    this.client = new MilvusClient(config);
  }

  async createDatabase({ name }: MilvusCreateDatabaseParams): Promise<void> {
    try {
      const res = await this.client.createDatabase({
        db_name: name,
      });
      if (res.code != 0) {
        throw new MastraError({
          id: createVectorErrorId('MILVUS', 'CREATE_DB', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { reason: res.reason, errorCode: res.error_code },
        });
      }
    } catch (error) {
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'CREATE_DB', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
        },
        error,
      );
    }
  }

  transformFilter(filter?: MilvusVectorFilter): string {
    const translator = new MilvusFilterTranslator();
    const translated = translator.translate(filter);
    // TODO
    return '';
  }

  async query({
    indexName,
    queryVector,
    topK = 10,
    filter,
    includeVector = false,
    partitions,
    sparseVector,
  }: MilvusQueryVectorParams): Promise<QueryResult[]> {
    const translatedFilter = this.transformFilter(filter) ?? undefined;
    try {
      let results: SearchResultData[] = [];
      if (queryVector) {
        if (sparseVector) {
          // TODO
        } else {
          const res = await this.client.search({
            collection_name: indexName,
            partition_names: partitions,
            filter: translatedFilter,
            vector: queryVector,
            topk: topK,
            output_fields: ['text', 'embedding'],
          });
          results = res.results;
        }
      } else {
        // TODO
        // const res = await this.client.query({
        //     collection_name: indexName,
        //     partition_names: partitions,
        //     filter: this.transformFilter(filter),
        //     output_fields: ["text", "embedding"]
        // });
        // results = res.data
      }
      return results.map(result => ({
        id: result.id,
        score: result.score,
        metadata: result,
        ...(includeVector && { vector: result['embedding'] }),
      }));
    } catch (error) {
      throw new MastraError(
        {
          id: createVectorErrorId('MILVUS', 'QUERY', 'FAILED'),
          domain: ErrorDomain.STORAGE,
          category: ErrorCategory.THIRD_PARTY,
          details: { indexName, topK, filter: JSON.stringify(filter) },
        },
        error,
      );
    }
  }

  async upsert({ indexName, vectors, metadata, ids, partition }: MilvusUpsertVectorParams): Promise<string[]> {
    validateUpsertInput('MILVUS', vectors, metadata, ids);

    // Generate IDs if not provided
    const vectorIds = ids || vectors.map(() => crypto.randomUUID());

    const records = vectors.map((vector, i) => ({
      id: vectorIds[i]!,
      embedding: vector,
      // TODO: Shold text field be provided explicitly? Do we even need it at all?   
      ...(metadata?.[i] ?? {}),
    }));

    try {
      for (let i = 0; i < records.length; i += BATCH_SIZE) {
        const batch = records.slice(i, i + BATCH_SIZE);
        await this.client.upsert({
          collection_name: indexName,
          partition_name: partition,
          data: batch,
        });
      }
      return vectorIds;
    } catch (error) {
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

  async createIndex({ indexName, dimension, metric = 'cosine', maxLength }: MilvusCreateIndexParams): Promise<void> {
    try {
      if (!Number.isInteger(dimension) || dimension <= 0) {
        throw new Error('Dimension must be a positive integer');
      }
      if (metric && !['cosine', 'euclidean', 'dotproduct'].includes(metric)) {
        throw new Error('Metric must be one of: cosine, euclidean, dotproduct');
      }
      if (maxLength <= 0) {
        throw new Error('Max length must be a positive integer');
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
            data_type: DataType.Int64,
            is_primary_key: true,
            autoID: false,
          },
          {
            name: 'embedding',
            data_type: DataType.FloatVector,
            dim: dimension,
          },
          {
            name: 'text',
            data_type: DataType.VarChar,
            max_length: maxLength,
          },
        ],
        dimension: dimension,
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
        field_name: 'embedding',
        metric_type: METRIC_MAPPING[metric],
        // TODO: Support other ANN algs
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
    } catch (error: any) {
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
   * Retrieves statistics about a vector index (collection).
   *
   * @param {string} indexName - The name of the index (collection) to describe
   * @returns A promise that resolves to the index statistics including dimension, count and metric
   */
  async describeIndex({ indexName }: DescribeIndexParams): Promise<MilvusIndexStats> {
    try {
      const stats = await this.client.getCollectionStatistics({
        collection_name: indexName,
      });

      const partitions = await this.client.showPartitions({
        collection_name: indexName,
      });

      const description = await this.client.describeCollection({
        collection_name: indexName,
      });

      // TODO: custom schema support
      const vectorField = description.schema.fields.find(f => f.name === 'embedding');

      if (!vectorField) throw new Error(`No embedding field for collection ${indexName}`);

      return {
        dimension: vectorField.dim as number,
        count: stats.data['row_count'],
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

  updateVector(params: UpdateVectorParams<MilvusVectorFilter>): Promise<void> {
    throw new Error('Method not implemented.');
  }

  /**
   * Deletes a vector by its ID.
   * @param indexName - The name of the index (collection) containing the vector.
   * @param id - The ID of the vector to delete.
   * @returns A promise that resolves when the deletion is complete.
   * @throws Will throw an error if the deletion operation fails.
   */
  async deleteVector({ indexName, id, partition }: MilvusDeleteVectorParams): Promise<void> {
    try {
      await this.client.delete({
        ids: [id],
        collection_name: indexName,
        partition_name: partition,
      });
    } catch (error) {
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
   * Deletes multiple vectors by IDs or filter.
   * @param indexName - The name of the index (collection) containing the vectors.
   * @param ids - Array of vector IDs to delete (mutually exclusive with filter).
   * @param filter - Filter to match vectors to delete (mutually exclusive with ids).
   * @param partition - The partition of the collection (optional, Milvus-specific).
   * @returns A promise that resolves when the deletion is complete.
   * @throws Will throw an error if both ids and filter are provided, or if neither is provided.
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
