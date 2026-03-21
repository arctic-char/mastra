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
import { MilvusClient, type ClientConfig } from "@zilliz/milvus2-sdk-node";

import { MilvusFilterTranslator } from './filter';
import type { MilvusVectorFilter } from './filter';


export type MilvusVectorConfig = ClientConfig & {
    /** The unique identifier for this vector store instance. */
    id: string;
};


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

    query(params: QueryVectorParams<MilvusVectorFilter>): Promise<QueryResult[]> {
        throw new Error('Method not implemented.');
    }

    upsert(params: UpsertVectorParams): Promise<string[]> {
        throw new Error('Method not implemented.');
    }

    createIndex(params: CreateIndexParams): Promise<void> {
        throw new Error('Method not implemented.');
    }

    listIndexes(): Promise<string[]> {
        throw new Error('Method not implemented.');
    }

    describeIndex(params: DescribeIndexParams): Promise<IndexStats> {
        throw new Error('Method not implemented.');
    }

    deleteIndex(params: DeleteIndexParams): Promise<void> {
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
