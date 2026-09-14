from backend.agents.gemini_client import GeminiClient


def main():

    print("=" * 60)
    print("GEMINI CONNECTION TEST")
    print("=" * 60)

    client = GeminiClient()

    response = client.generate(
        prompt=(
            "Explain in one sentence what RAG means."
        ),
        system_instruction=(
            "You are a helpful assistant. "
            "Answer briefly."
        ),
    )

    print()
    print("GEMINI RESPONSE:")
    print(response)

    print()
    print("=" * 60)
    print("GEMINI TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()