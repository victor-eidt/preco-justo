import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Run the Preço Justo demo app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run("precojusto.app.server:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
