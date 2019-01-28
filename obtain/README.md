# Obtain

## Run SQL Queries

```python
pd.read_sql_query(
    sql="""
    SELECT genus, count(*)
    FROM taxa
    WHERE isExtinct = 0.0
      AND genus IS NOT NULL
    GROUP BY 1
    ORDER BY 2 DESC
    LIMIT 10
    """,
    con=conn
)
```

## If you make changes a the DB, save them

```python
conn.commit()
conn.close()
```

## Improve Speed With a New Index

If you know you will be pulling records according to the value of a certain column(s) very frequently, make a new index for your database on that column.
In the example below, we're setting the id column as the new and assigning the name id_idx to it.

```python
curs.execute("CREATE INDEX id_idx ON data (id);")
conn.commit()
```
