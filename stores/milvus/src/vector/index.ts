import { MastraError, ErrorDomain, ErrorCategory } from '@mastra/core/error';
import { createVectorErrorId } from '@mastra/core/storage';
import { MastraVector } from '@mastra/core/vector';
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
import { MilvusClient, type ClientConfig, DataType } from '@zilliz/milvus2-sdk-node';

import { MilvusFilterTranslator } from './filter';
import type { MilvusVectorFilter } from './filter';

const METRIC_MAPPING: Record<string, string> = {
  cosine: 'COSINE',
  euclidean: 'L2',
  dotproduct: 'IP',
};

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

  async query(params: QueryVectorParams<MilvusVectorFilter>): Promise<QueryResult[]> {
    throw new Error('Method not implemented.');
  }

  async upsert(params: UpsertVectorParams): Promise<string[]> {
    throw new Error('Method not implemented.');
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
            max_length: 1024,
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
    throw new Error('Method not implemented.');
  }

  async describeIndex({ indexName }: DescribeIndexParams): Promise<IndexStats> {
    throw new Error('Method not implemented.');
  }

  async deleteIndex({ indexName }: DeleteIndexParams): Promise<void> {
    throw new Error('Method not implemented.');
  }

  updateVector(params: UpdateVectorParams<MilvusVectorFilter>): Promise<void> {
    throw new Error('Method not implemented.');
  }

  deleteVector(params: DeleteVectorParams): Promise<void> {
    throw new Error('Method not implemented.');
  }

  deleteVectors(params: DeleteVectorsParams<MilvusVectorFilter>): Promise<void> {
    throw new Error('Method not implemented.');
  }
}
