# Query DSL

The :mod:`quantx.query.tsql` module implements a lightweight domain specific
language for filtering and aggregating time-series data.

## Grammar

```
WHERE    := <expr> [ (and|or) <expr> ]*
<expr>   := <column> <op> <value>
<op>     := in | == | != | > | < | >= | <=
<value>  := number | 'string' | ( 'str1', 'str2', ... )
```

The ``in`` operator accepts a parenthesised list which maps to a Python list
internally. Expressions are evaluated using :func:`pandas.DataFrame.query`.

## Examples

```python
from quantx.query import tsql
filtered = tsql.where(df, "symbol in ('A','B') and price > 10")
agg = tsql.aggregate(filtered, ['symbol', 'window'], {
    'price': ['sum', 'mean', 'ohlc', 'last'],
    'volume': ['sum']
})
ema = tsql.rolling(df, 20, 'ema', column='price')
```

## Performance notes

The DSL is intentionally small and compiles to pandas operations. For large
frames consider pre-filtering columns and minimising the number of Python
objects created during evaluation.
