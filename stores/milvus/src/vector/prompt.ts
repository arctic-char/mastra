/**
 * Vector store specific prompt that details supported operators and examples.
 * This prompt helps users construct valid filters for Milvus Vector.
 */
export const MILVUS_PROMPT = `When querying Milvus via this integration, you can ONLY use the operators listed below. Any other operators will be rejected.
Important: Don't explain how to construct the filter - use the specified operators and fields to search the content and return relevant results.
If a user tries to give an explicit operator that is not supported, reject the filter entirely and let them know that the operator is not supported.

Filters apply to the \`metadata\` JSON field. Each top-level key is a metadata key segment (letters, digits, underscore only). Use nested objects for nested metadata paths — do not use dot-separated keys like "user.name".

Basic Comparison Operators:
- $eq: Exact match (default when using field: value)
  Example: { "category": "electronics" }
- $ne: Not equal
  Example: { "category": { "$ne": "electronics" } }
- $gt: Greater than
  Example: { "price": { "$gt": 100 } }
- $gte: Greater than or equal
  Example: { "price": { "$gte": 100 } }
- $lt: Less than
  Example: { "price": { "$lt": 100 } }
- $lte: Less than or equal
  Example: { "price": { "$lte": 100 } }

Array Operators:
- $in: Match any value in array (non-empty array required)
  Example: { "category": { "$in": ["electronics", "books"] } }
- $nin: Does not match any value in array (non-empty array required)
  Example: { "category": { "$nin": ["electronics", "books"] } }
- $all: All listed values must appear in a JSON array field (uses JSON_CONTAINS_ALL; non-empty array required)
  Example: { "tags": { "$all": ["premium", "sale"] } }
- Direct array value on a field: treated as membership in that list (non-empty)
  Example: { "tags": ["a", "b"] }

Logical Operators:
- $and: Logical AND (implicit when multiple top-level field conditions are combined)
  Implicit Example: { "price": { "$gt": 100 }, "category": "electronics" }
  Explicit Example: { "$and": [{ "price": { "$gt": 100 } }, { "category": "electronics" }] }
- $or: Logical OR
  Example: { "$or": [{ "price": { "$lt": 50 } }, { "category": "books" }] }
- $not: Logical NOT (object value, non-empty)
  Example: { "$not": { "archived": true } }
  Field-level Example: { "age": { "$not": { "$gte": 18 } } }
- $nor: Logical NOR (non-empty array of condition objects)
  Example: { "$nor": [{ "a": 1 }, { "b": 2 }] }

Element Operators:
- $exists: Field presence in metadata (boolean)
  Example: { "rating": { "$exists": true } }

Nested metadata paths (object form, not dots):
  Example: { "user": { "name": "Ada" } }

Unsupported (will be rejected or throw):
- $regex, $options, $elemMatch, $size, $contains
- RegExp literal values
- Empty arrays as field values; empty objects as field values; empty $in / $nin / $all
- Field key segments that are not [a-zA-Z_][a-zA-Z0-9_]* (no dots in a single key — nest instead)

Restrictions:
- Only logical operators ($and, $or, $not, $nor) may appear at the root alongside field keys; other operators must be inside a field condition.
  Valid: { "field": { "$gt": 100 } }
  Valid: { "$and": [...] }
  Invalid: { "$gt": 100 }
- Logical operator arrays must be non-empty where required; each $and/$or/$nor element must be an object.
- Logical arrays must contain field conditions, not bare operators.
  Valid: { "$and": [{ "field": { "$gt": 100 } }] }
  Invalid: { "$and": [{ "$gt": 100 }] }
- $and, $or, $nor: use at top level or nested under other logical operators — not as a value inside an arbitrary field object mixed with non-operator keys.
- $not at root must be a non-empty object; at field level must be an object of operator conditions.
- Dates in filters are normalized to ISO-8601 strings for comparisons.
- null can be used for equality on a field.

Example Complex Query:
{
  "$and": [
    { "category": { "$in": ["electronics", "computers"] } },
    { "price": { "$gte": 100, "$lte": 1000 } },
    { "tags": { "$all": ["premium"] } },
    { "rating": { "$exists": true } },
    { "$or": [
      { "stock": { "$gt": 0 } },
      { "preorder": true }
    ]}
  ]
}`;
