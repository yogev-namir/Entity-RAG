def basic_augment():
    for query in APP_QUERIES:
        retrieved_docs = retrieve_from_index(query)
        print(retrieved_docs)
        source_knowledge = "\n\n".join(retrieved_docs[0])

        augmented_prompt = (
            f"Using the contexts below, answer the query. If you don't know the answer, say you don't know."
            f"Source: {source_knowledge}"
            f"Query: {query}"
        )
        response = co.chat(
            model='command-r-plus',
            message=augmented_prompt,
        )
        print(response.text)


def augment_prompt1(query: str, top_relevant_docs: list) -> str:
    """
    Augment the prompt with the top 3 results from the knowledge base
    Args:
        query: The query to augment
        top_relevant_docs: The k-retrived docs
    Returns:
        str: The augmented prompt
    """
    symptoms_matches = set()
    diseases_matches = set()
    cases_match = []
    for doc in top_relevant_docs:
        if 'SIGN_SYMPTOM' in doc.keys():
            symptoms_matches.update(doc["SIGN_SYMPTOM"])
        if 'DISEASE_DISORDER' in doc.keys():
            diseases_matches.update(doc['DISEASE_DISORDER'])
        cases_match.append(doc["text"])

    source_knowledge = "\n\n".join(cases_match)

    augmented_prompt = (
        f"Using the contexts below, answer the query while aligning with the medical information provided.\n\n"
        f"Related Symptoms: {', '.join(symptoms_matches)}\n\n"
        f"Related Disorders: {', '.join(diseases_matches)}\n\n"
        f"Some medical cases you might find relevant:\n{source_knowledge}\n\n"
        f"Query: {query}"
    )
    return augmented_prompt, symptoms_matches, diseases_matches, source_knowledge


def augment_prompt2(query: str, top_relevant_docs: list, options: dict) -> str:
    """
    Augment the prompt with the top relevant results, including symptoms, diseases, and multiple-choice options.
    Args:
        query: The question to answer.
        top_relevant_docs: The k-retrieved documents.
        options: Dictionary of options (e.g., {"opa": "17-46% of cases", "opb": "5-10 % of cases", ...}).
    Returns:
        str: The augmented prompt for the RAG system.
    """
    symptoms_matches = set()
    diseases_matches = set()
    cases_match = []

    # Extract relevant entities (symptoms, diseases) and case details in one loop
    for doc in top_relevant_docs:
        symptoms_matches.update(doc.get("SIGN_SYMPTOM", []))
        diseases_matches.update(doc.get("DISEASE_DISORDER", []))
        cases_match.append(doc.get("text", ""))

    options_text = "\n".join(f"{key}: {value}" for key, value in options.items())
    source_knowledge = "\n\n".join(cases_match)

    augmented_prompt = (
        f"Using the contexts below, determine the correct answer to the question. "
        f"Then explain your choice based on the provided context.\n\n"
        f"Question: {query}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Related Symptoms: {', '.join(symptoms_matches) or 'N/A'}\n\n"
        f"Related Disorders: {', '.join(diseases_matches) or 'N/A'}\n\n"
        f"Relevant Contexts:\n{source_knowledge}\n\n"
        f"Your response should follow this format:\n"
        f"<correct answer from [opa, opb, opc, opd]>.\n"
        f"Explanation: <relevant information from the context>."
    )

    return augmented_prompt


def generate_response(augmented_prompt):
    response = co.chat(
        model='command-r-plus',
        message=augmented_prompt,
    )
    if not response.text or "cannot" in response.text.lower():
        explanation = ("Explanation: The retrieved knowledge did not contain sufficient context to determine the "
                       "correct answer.")
        return f"Cannot determine the correct answer.\n{explanation}"

    print("==============================================================================================")
    print(response.text)
    return response.text


def start():
    path = "../src/data/medmcqa/test.json"
    test_set = pd.read_json(path)


def generate_response(query, metadatas):
    augmented_prompt, symptoms_matches, diseases_matches, source_knowledge = augment_prompt(query, metadatas)
    print(f"{top_relevant_docs}")
    response = co.chat(
        model='command-r-plus',
        message=augmented_prompt,
    )
    if correct_option is None:
        explanation = f"Explanation: The retrieved knowledge did not contain sufficient context to determine the correct answer."
        return f"Cannot determine the correct answer.\n{explanation}"
    print("==============================================================================================")
    print(response.text)
    print("==============================================================================================")


