import { beforeEach, describe, expect, it } from 'vitest';

import type { MilvusVectorFilter } from './filter';
import { MilvusFilterTranslator } from './filter';

describe('MilvusFilterTranslator', () => {
  let t: MilvusFilterTranslator;

  beforeEach(() => {
    t = new MilvusFilterTranslator();
  });

  describe('empty and trivial', () => {
    it('should return empty string for undefined, null, or empty object', () => {
      expect(t.translate(undefined)).toBe('');
      expect(t.translate(null as unknown as MilvusVectorFilter)).toBe('');
      expect(t.translate({})).toBe('');
    });
  });

  describe('implicit equality and paths', () => {
    it('should translate implicit equality on metadata JSON path', () => {
      expect(t.translate({ category: 'books' })).toBe('metadata["category"] == "books"');
      expect(t.translate({ count: 3 })).toBe('metadata["count"] == 3');
      expect(t.translate({ active: true })).toBe('metadata["active"] == true');
    });

    it('should translate null equality', () => {
      expect(t.translate({ absent: null })).toBe('metadata["absent"] == null');
    });

    it('should translate implicit AND of multiple fields', () => {
      const expr = t.translate({ category: 'A', n: 2 });
      expect(expr).toBe('(metadata["category"] == "A") && (metadata["n"] == 2)');
    });

    it('should translate array field value as in list', () => {
      expect(t.translate({ tags: ['a', 'b'] })).toBe('metadata["tags"] in ["a", "b"]');
    });

    it('should translate nested object paths', () => {
      expect(t.translate({ user: { name: 'Ada' } } as MilvusVectorFilter)).toBe(
        'metadata["user"]["name"] == "Ada"',
      );
    });

    it('should translate deep single-leaf nesting without redundant wrapping', () => {
      expect(t.translate({ a: { b: { c: 1 } } } as MilvusVectorFilter)).toBe(
        'metadata["a"]["b"]["c"] == 1',
      );
    });

    it('should allow underscore in field segments', () => {
      expect(t.translate({ _id: 'x' })).toBe('metadata["_id"] == "x"');
    });

    it('should escape quotes and backslashes in string literals', () => {
      expect(t.translate({ msg: 'say "hi"' })).toBe('metadata["msg"] == "say \\"hi\\""');
      expect(t.translate({ p: 'a\\b' })).toBe('metadata["p"] == "a\\\\b"');
    });
  });

  describe('operators on fields', () => {
    it('should translate comparison operators', () => {
      expect(t.translate({ price: { $gt: 10 } })).toBe('metadata["price"] > 10');
      expect(t.translate({ price: { $lt: 5 } })).toBe('metadata["price"] < 5');
      expect(t.translate({ price: { $gte: 10, $lte: 20 } })).toBe(
        '(metadata["price"] >= 10) && (metadata["price"] <= 20)',
      );
    });

    it('should translate $ne', () => {
      expect(t.translate({ status: { $ne: 'done' } })).toBe('metadata["status"] != "done"');
    });

    it('should translate $eq explicitly', () => {
      expect(t.translate({ k: { $eq: 7 } })).toBe('metadata["k"] == 7');
    });

    it('should translate $in and $nin', () => {
      expect(t.translate({ id: { $in: ['a', 'b'] } })).toBe('metadata["id"] in ["a", "b"]');
      expect(t.translate({ id: { $nin: [1, 2] } })).toBe('not (metadata["id"] in [1, 2])');
    });

    it('should translate $all with JSON_CONTAINS_ALL', () => {
      expect(t.translate({ tags: { $all: ['x', 'y'] } })).toBe(
        'JSON_CONTAINS_ALL(metadata["tags"], ["x", "y"])',
      );
    });

    it('should translate $exists', () => {
      expect(t.translate({ key: { $exists: true } })).toBe('exists(metadata["key"])');
      expect(t.translate({ key: { $exists: false } })).toBe('not (exists(metadata["key"]))');
    });

    it('should translate operators on nested metadata paths', () => {
      expect(t.translate({ user: { age: { $lt: 18 } } } as MilvusVectorFilter)).toBe(
        'metadata["user"]["age"] < 18',
      );
    });

    it('should translate field-level $not with multiple inner operators', () => {
      expect(t.translate({ age: { $not: { $gte: 18, $lte: 65 } } })).toBe(
        'not ((metadata["age"] >= 18) && (metadata["age"] <= 65))',
      );
    });
  });

  describe('logical operators', () => {
    it('should translate $and and $or', () => {
      expect(
        t.translate({
          $and: [{ a: 1 }, { b: 2 }],
        }),
      ).toBe('(metadata["a"] == 1) && (metadata["b"] == 2)');

      expect(
        t.translate({
          $or: [{ status: 'open' }, { status: 'pending' }],
        }),
      ).toBe('(metadata["status"] == "open") || (metadata["status"] == "pending")');
    });

    it('should translate $and with a single clause', () => {
      expect(t.translate({ $and: [{ x: 1 }] })).toBe('(metadata["x"] == 1)');
    });

    it('should translate nested logical operators', () => {
      expect(
        t.translate({
          $and: [{ t: 1 }, { $or: [{ u: 2 }, { v: 3 }] }],
        }),
      ).toBe('(metadata["t"] == 1) && ((metadata["u"] == 2) || (metadata["v"] == 3))');
    });

    it('should translate $not', () => {
      expect(t.translate({ $not: { archived: true } })).toBe('not (metadata["archived"] == true)');
    });

    it('should translate $nor', () => {
      expect(
        t.translate({
          $nor: [{ a: 1 }, { b: 2 }],
        }),
      ).toBe('not (metadata["a"] == 1) && not (metadata["b"] == 2)');
    });

    it('should translate field-level $not', () => {
      expect(t.translate({ age: { $not: { $gte: 18 } } })).toBe('not (metadata["age"] >= 18)');
    });

    it('should combine top-level field conditions with logical operators', () => {
      const expr = t.translate({
        kind: 'post',
        $or: [{ public: true }, { owner: 'me' }],
      });
      expect(expr).toBe(
        '(metadata["kind"] == "post") && ((metadata["public"] == true) || (metadata["owner"] == "me"))',
      );
    });
  });

  describe('dates and normalization', () => {
    it('should normalize Date to ISO strings for implicit equality', () => {
      const d = new Date('2024-01-15T00:00:00.000Z');
      expect(t.translate({ created: d })).toBe('metadata["created"] == "2024-01-15T00:00:00.000Z"');
    });

    it('should normalize Date inside operator objects', () => {
      const d = new Date('2024-06-01T00:00:00.000Z');
      expect(t.translate({ created: { $gte: d } })).toBe(
        'metadata["created"] >= "2024-06-01T00:00:00.000Z"',
      );
    });

    it('should normalize dates inside $in lists', () => {
      const d = new Date('2024-01-01T00:00:00.000Z');
      expect(t.translate({ day: { $in: [d] } })).toBe(
        'metadata["day"] in ["2024-01-01T00:00:00.000Z"]',
      );
    });
  });

  describe('unsupported operators (validation)', () => {
    it('should reject $regex, $options, and $elemMatch', () => {
      expect(() => t.translate({ x: { $regex: '.*' } } as MilvusVectorFilter)).toThrow(
        /Unsupported operator: \$regex/,
      );
      expect(() => t.translate({ x: { $options: 'i' } } as MilvusVectorFilter)).toThrow(
        /Unsupported operator: \$options/,
      );
      expect(() => t.translate({ x: { $elemMatch: { $gt: 1 } } } as MilvusVectorFilter)).toThrow(
        /Unsupported operator: \$elemMatch/,
      );
    });

    it('should reject field-only operators at root', () => {
      expect(() => t.translate({ $eq: 1 } as unknown as MilvusVectorFilter)).toThrow(
        /Invalid top-level operator: \$eq/,
      );
    });

    it('should reject unsupported custom operators such as $size', () => {
      expect(() => t.translate({ tags: { $size: 2 } } as unknown as MilvusVectorFilter)).toThrow(
        /Unsupported operator: \$size/,
      );
    });
  });

  describe('invalid structure and values', () => {
    it('should reject invalid field segments', () => {
      expect(() => t.translate({ 'bad-seg': 1 })).toThrow(/Invalid metadata field segment/);
      expect(() => t.translate({ '0bad': 1 })).toThrow(/Invalid metadata field segment/);
    });

    it('should reject regex values', () => {
      expect(() => t.translate({ f: /x/ as unknown as string })).toThrow(/Regex/);
    });

    it('should reject non-finite numbers', () => {
      expect(() => t.translate({ n: Infinity })).toThrow(/Non-finite numbers/);
      expect(() => t.translate({ n: { $gt: NaN } })).toThrow(/Non-finite numbers/);
    });

    it('should reject unsupported literal types', () => {
      expect(() => t.translate({ n: { $eq: 1n as unknown as number } })).toThrow(
        /Unsupported literal type/,
      );
    });

    it('should reject empty array as direct field value', () => {
      expect(() => t.translate({ tags: [] })).toThrow(/Empty array is not a valid filter value/);
    });

    it('should reject empty object as field value', () => {
      expect(() => t.translate({ meta: {} })).toThrow(/Empty object is not a valid filter value/);
    });

    it('should reject empty nested array', () => {
      expect(() => t.translate({ user: { ids: [] } } as MilvusVectorFilter)).toThrow(
        /Empty array is not a valid nested filter value/,
      );
    });

    it('should reject empty $in, $nin, and $all', () => {
      expect(() => t.translate({ x: { $in: [] } })).toThrow(/\$in requires a non-empty array/);
      expect(() => t.translate({ x: { $nin: [] } })).toThrow(/\$nin requires a non-empty array/);
      expect(() => t.translate({ x: { $all: [] } })).toThrow(/\$all requires a non-empty array/);
    });

    it('should reject non-boolean $exists', () => {
      expect(() => t.translate({ x: { $exists: 1 as unknown as boolean } })).toThrow(
        /\$exists value must be a boolean/,
      );
    });

    it('should reject empty or invalid logical arrays', () => {
      expect(() => t.translate({ $and: [] })).toThrow(/\$and requires a non-empty array/);
      expect(() => t.translate({ $or: [] })).toThrow(/\$or requires a non-empty array/);
      expect(() => t.translate({ $nor: [] })).toThrow(/\$nor requires a non-empty array/);
      expect(() => t.translate({ $and: [1] as unknown as { a: number }[] })).toThrow(
        /\$and\[0\] must be an object/,
      );
    });

    it('should reject invalid $not values', () => {
      expect(() => t.translate({ $not: [] as unknown as Record<string, unknown> })).toThrow(
        /\$not operator requires an object/,
      );
      expect(() => t.translate({ $not: {} })).toThrow(/\$not operator cannot be empty/);
      expect(() => t.translate({ $not: 1 as unknown as Record<string, unknown> })).toThrow(
        /\$not operator requires an object/,
      );
    });

    it('should reject $not at translate time when value is not an object', () => {
      expect(() =>
        t.translate({ $not: ['x'] as unknown as Record<string, unknown> }),
      ).toThrow(/\$not operator requires an object/);
    });

    it('should reject field-level $not with array payload (same validation as root $not)', () => {
      expect(() => t.translate({ age: { $not: [] as unknown as Record<string, unknown> } })).toThrow(
        /\$not operator requires an object/,
      );
    });
  });
});
