import { BaseFilterTranslator } from '@mastra/core/vector/filter';
import type {
  VectorFilter,
  OperatorSupport,
  OperatorValueMap,
  LogicalOperatorValueMap,
  BlacklistedRootOperators,
  QueryOperator,
  FilterValue,
  OperatorCondition,
} from '@mastra/core/vector/filter';

type MilvusOperatorValueMap = OperatorValueMap;

type MilvusLogicalOperatorValueMap = LogicalOperatorValueMap;

type MilvusBlacklisted = BlacklistedRootOperators;

export type MilvusVectorFilter = VectorFilter<
  keyof MilvusOperatorValueMap,
  MilvusOperatorValueMap,
  MilvusLogicalOperatorValueMap,
  MilvusBlacklisted
>;

export class MilvusFilterTranslator extends BaseFilterTranslator<MilvusVectorFilter> {
    
    translate(filter: MilvusVectorFilter): MilvusVectorFilter {
        throw new Error('Method not implemented.');
    }
}
