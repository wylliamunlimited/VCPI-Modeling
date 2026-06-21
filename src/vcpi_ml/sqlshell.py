"""Interactive SQL Shell over vcpi.query (remote, no download)

Just for convenience - no more Pythonic querying, work purely on SQL

Run Command:
`uv run --env-file .env python src/vcpi_ml/sqlshell.py`"""

import vcpi

def main():

    job = None

    print("= = = = = = = = = = VCPI SQL SHELL (Ctrl-D or 'quit' to exit.) = = = = = = = = = =")

    while True:

        try:
            sql = input("sql> ").strip()
        except EOFError:
            break
        
        if not sql or sql.lower() in {"quit", "exit"}:
            print("= = = = = = = Shutting down = = = = = = =")
            break

        try:
            df = vcpi.query(job=job, sql=sql)
            print(df)
        except Exception as e:
            print("ERROR: ", e)

if __name__ == "__main__":
    main()