import { BaseFilterTranslator } from '@mastra/core/vector/filter';
import type {
  VectorFilter,
  OperatorSupport,
  OperatorValueMap,
  LogicalOperatorValueMap,
  BlacklistedRootOperators,
  QueryOperator,
} from '@mastra/core/vector/filter';

type MilvusOperatorValueMap = Omit<OperatorValueMap, '$regex' | '$options' | '$elemMatch'>;

type MilvusLogicalOperatorValueMap = Pick<
  LogicalOperatorValueMap,
  '$and' | '$or' | '$not' | '$nor'
>;

type MilvusBlacklisted = BlacklistedRootOperators;

/**
 * Mongo-style filters for Milvus collections whose dynamic attributes live in the `metadata` JSON column.
 *
 * @remarks
 * Output is a Milvus **boolean expression** string (see
 * [Scalar filtering](https://milvus.io/docs/boolean.md) and
 * [JSON fields](https://milvus.io/docs/use-json-fields.md)).
 *
 * - Field `foo` → `metadata["foo"]`. Nested keys use `metadata["a"]["b"]`.
 * - Logical: `&&`, `||`, `not (...)`.
 * - Lists: lowercase `in` / `not (x in [...])`.
 * - JSON array membership for `$all`: `JSON_CONTAINS_ALL(metadata["tags"], [...])`.
 * - Key presence: `$exists` → `exists(metadata["key"])` / `not exists(metadata["key"])`.
 *
 * Not supported here: `$regex` / `$options`, `$elemMatch`, `$size`, `$contains` (enforced in types via
 * {@link MilvusOperatorValueMap} and {@link MilvusFilterTranslator.getSupportedOperators}).
 */
export type MilvusVectorFilter = VectorFilter<
  keyof MilvusOperatorValueMap,
  MilvusOperatorValueMap,
  MilvusLogicalOperatorValueMap,
  MilvusBlacklisted
>;

const FIELD_SEGMENT = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

export class MilvusFilterTranslator extends BaseFilterTranslator<MilvusVectorFilter, string> {
  protected override getSupportedOperators(): OperatorSupport {
    return {
      logical: ['$and', '$or', '$not', '$nor'],
      basic: ['$eq', '$ne'],
      numeric: ['$gt', '$gte', '$lt', '$lte'],
      array: ['$in', '$nin', '$all'],
      element: ['$exists'],
      regex: [],
      custom: [],
    };
  }

  translate(filter?: MilvusVectorFilter): string {
    if (this.isEmpty(filter)) {
      return '';
    }
    this.validateFilter(filter);
    return this.buildExpression(filter as Record<string, unknown>);
  }

  private buildExpression(node: Record<string, unknown>): string {
    const parts: string[] = [];

    for (const [key, value] of Object.entries(node)) {
      if (this.isLogicalOperator(key)) {
        parts.push(this.translateLogical(key as QueryOperator, value));
      } else {
        parts.push(this.translateField(key, value));
      }
    }

    if (parts.length === 0) {
      return '';
    }
    if (parts.length === 1) {
      return parts[0]!;
    }
    return parts.map(p => this.wrap(p)).join(' && ');
  }

  private translateLogical(operator: QueryOperator, value: unknown): string {
    if (operator === '$and') {
      const clauses = this.normalizeLogicalArray(value, '$and');
      return clauses.map(c => this.wrap(this.buildExpression(c))).join(' && ');
    }
    if (operator === '$or') {
      const clauses = this.normalizeLogicalArray(value, '$or');
      return clauses.map(c => this.wrap(this.buildExpression(c))).join(' || ');
    }
    if (operator === '$not') {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error('$not requires a non-array object value');
      }
      const inner = this.buildExpression(value as Record<string, unknown>);
      return `not (${inner})`;
    }
    if (operator === '$nor') {
      const clauses = this.normalizeLogicalArray(value, '$nor');
      if (clauses.length === 0) {
        throw new Error('$nor requires a non-empty array');
      }
      return clauses.map(c => `not (${this.buildExpression(c)})`).join(' && ');
    }
    throw new Error(`Unsupported logical operator: ${operator}`);
  }

  private normalizeLogicalArray(value: unknown, name: string): Record<string, unknown>[] {
    if (!Array.isArray(value) || value.length === 0) {
      throw new Error(`${name} requires a non-empty array`);
    }
    return value.map((item, i) => {
      if (typeof item !== 'object' || item === null || Array.isArray(item)) {
        throw new Error(`${name}[${i}] must be an object`);
      }
      return item as Record<string, unknown>;
    });
  }

  private translateField(field: string, value: unknown): string {
    this.assertValidFieldSegment(field);
    const path = this.metadataPath([field]);

    if (value instanceof Date) {
      return `${path} == ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
    }

    if (this.isPrimitive(value)) {
      return `${path} == ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
    }
    if (this.isRegex(value)) {
      throw new Error('Regex filters are not supported for Milvus metadata expressions');
    }
    if (Array.isArray(value)) {
      if (value.length === 0) {
        throw new Error('Empty array is not a valid filter value; use $in with an explicit list');
      }
      return `${path} in ${this.formatListLiteral(this.normalizeArrayValues(value))}`;
    }
    if (typeof value === 'object' && value !== null) {
      const obj = value as Record<string, unknown>;
      const keys = Object.keys(obj);
      if (keys.length === 0) {
        throw new Error(`Empty object is not a valid filter value for field "${field}"`);
      }
      const allOperators = keys.every(k => this.isOperator(k));
      if (allOperators) {
        return this.translateFieldOperators(path, obj);
      }
      return this.translateNestedObject([field], obj);
    }
    throw new Error(`Unsupported filter value for field "${field}"`);
  }

  private translateFieldOperators(jsonPath: string, ops: Record<string, unknown>): string {
    const entries = Object.entries(ops);
    if (entries.length === 0) {
      throw new Error('Operator object cannot be empty');
    }

    const parts: string[] = [];

    for (const [op, raw] of entries) {
      if (op === '$not') {
        if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
          throw new Error('$not requires an object of operator conditions');
        }
        parts.push(`not (${this.translateFieldOperators(jsonPath, raw as Record<string, unknown>)})`);
        continue;
      }

      if (!this.isOperator(op)) {
        throw new Error(`Unknown operator key "${op}" in field filter`);
      }

      parts.push(this.translateScalarOperator(jsonPath, op as QueryOperator, raw));
    }

    if (parts.length === 1) {
      return parts[0]!;
    }
    return parts.map(p => this.wrap(p)).join(' && ');
  }

  private translateScalarOperator(path: string, operator: QueryOperator, value: unknown): string {
    switch (operator) {
      case '$eq':
        return `${path} == ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$ne':
        return `${path} != ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$gt':
        return `${path} > ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$gte':
        return `${path} >= ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$lt':
        return `${path} < ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$lte':
        return `${path} <= ${this.formatLiteral(this.normalizeComparisonValue(value))}`;
      case '$in': {
        if (!Array.isArray(value) || value.length === 0) {
          throw new Error('$in requires a non-empty array');
        }
        return `${path} in ${this.formatListLiteral(this.normalizeArrayValues(value))}`;
      }
      case '$nin': {
        if (!Array.isArray(value) || value.length === 0) {
          throw new Error('$nin requires a non-empty array');
        }
        return `not (${path} in ${this.formatListLiteral(this.normalizeArrayValues(value))})`;
      }
      case '$all': {
        if (!Array.isArray(value) || value.length === 0) {
          throw new Error('$all requires a non-empty array');
        }
        const vals = this.normalizeArrayValues(value);
        return `JSON_CONTAINS_ALL(${path}, ${this.formatListLiteral(vals)})`;
      }
      case '$exists': {
        if (typeof value !== 'boolean') {
          throw new Error('$exists value must be a boolean');
        }
        return value ? `exists(${path})` : `not (exists(${path}))`;
      }
      default:
        throw new Error(`Unsupported operator "${operator}" for Milvus metadata expressions`);
    }
  }

  /** Nested object without operators → AND of leaf equalities under metadata path. */
  private translateNestedObject(segments: string[], obj: Record<string, unknown>): string {
    const parts: string[] = [];
    for (const [key, value] of Object.entries(obj)) {
      this.assertValidFieldSegment(key);
      const next = [...segments, key];
      const path = this.metadataPath(next);

      if (this.isPrimitive(value)) {
        parts.push(`${path} == ${this.formatLiteral(this.normalizeComparisonValue(value))}`);
      } else if (Array.isArray(value)) {
        if (value.length === 0) {
          throw new Error('Empty array is not a valid nested filter value');
        }
        parts.push(`${path} in ${this.formatListLiteral(this.normalizeArrayValues(value))}`);
      } else if (typeof value === 'object' && value !== null) {
        const child = value as Record<string, unknown>;
        const keys = Object.keys(child);
        const allOperators = keys.length > 0 && keys.every(k => this.isOperator(k));
        if (allOperators) {
          parts.push(this.translateFieldOperators(path, child));
        } else {
          parts.push(this.translateNestedObject(next, child));
        }
      } else {
        throw new Error(`Unsupported nested value under "${segments.join('.')}"`);
      }
    }
    if (parts.length === 1) {
      return parts[0]!;
    }
    return parts.map(p => this.wrap(p)).join(' && ');
  }

  private metadataPath(segments: string[]): string {
    return `metadata${segments.map(s => `["${this.escapeBracketSegment(s)}"]`).join('')}`;
  }

  private assertValidFieldSegment(segment: string): void {
    if (!FIELD_SEGMENT.test(segment)) {
      throw new Error(
        `Invalid metadata field segment "${segment}". Use letters, digits, and underscore only.`,
      );
    }
  }

  /** Escape double quotes inside JSON bracket path segments. */
  private escapeBracketSegment(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  }

  private formatLiteral(value: unknown): string {
    if (value === null) {
      return 'null';
    }
    if (typeof value === 'boolean') {
      return value ? 'true' : 'false';
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) {
        throw new Error('Non-finite numbers are not allowed in Milvus filter expressions');
      }
      return String(value);
    }
    if (typeof value === 'string') {
      return `"${this.escapeString(value)}"`;
    }
    if (value instanceof Date) {
      return `"${this.escapeString(value.toISOString())}"`;
    }
    throw new Error(`Unsupported literal type for Milvus expression: ${typeof value}`);
  }

  private formatListLiteral(values: unknown[]): string {
    return `[${values.map(v => this.formatLiteral(v)).join(', ')}]`;
  }

  private escapeString(s: string): string {
    return s.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  }

  private wrap(expr: string): string {
    return `(${expr})`;
  }
}
