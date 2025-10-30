import os
from src.ingest import main as ingest_main
from src.query import main as query_main
from src.eval_ragas import main as eval_main

BANNER = """
RAG PoC (OSS) Test Client
1) Ingest ./data
2) Ask a question
3) Evaluate with ./eval/questions.csv
4) Exit
"""

def run():
    while True:
        print(BANNER)
        choice = input("Select: ").strip()
        if choice == "1":
            ingest_main("./data")
        elif choice == "2":
            q = input("Question: ").strip()
            if q:
                query_main(q)
        elif choice == "3":
            eval_main("./eval/questions.csv")
        elif choice == "4":
            print("Bye.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    run()
