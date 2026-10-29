"""
==========================================================
Main Entry Point — Multi-Agent RAG System (CLI)
==========================================================
"""

import sys

import requests

from graph.workflow import MultiAgentSystem
from utils.logger import get_logger

logger = get_logger("main")


def main():
    try:
        logger.info("Starting Multi-Agent RAG System...")

        system = MultiAgentSystem()
        system.initialize()

        print("=" * 80)
        print("Multi-Agent RAG Assistant — type 'exit' to quit")
        print("Agents: Router, Planner, Retrieval, QA, Summarizer, Comparison, "
              "Table, Code, Image, Verifier, Memory")
        print("=" * 80)

        while True:
            question = input("\nYou : ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue

            result = system.ask(question)
            print(f"\n[{result.get('route')}] Answer:\n{result['answer']}\n")

            if result.get("sources"):
                print("Sources:")
                for s in result["sources"]:
                    print(f" - {s}")

            v = result.get("verification")
            if v and v.get("confidence") is not None:
                print(f"\nConfidence: {v['confidence']:.2f}")
                if v.get("unsupported_claims"):
                    print(f"⚠ Possibly unsupported: {v['unsupported_claims']}")

        print("\nGoodbye!")

    except requests.exceptions.Timeout:
        logger.error("Request to NVIDIA NIM timed out.")
        sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network error: {e}")
        sys.exit(1)
    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Check your .env file for NVIDIA_API_KEY and SERPER_API_KEY.")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Data error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
