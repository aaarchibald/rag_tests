# from llama_index.llms.ollama import Ollama

# llm = Ollama(model="llama3.2:latest", request_timeout=120.0)

# def query_ollama(prompt: str) -> str:
#     return llm.complete(prompt).text



# def groundedness_with_ollama(query, context, response):
#     prompt = f"""
# Given the following context:

# {context}

# And the question:
# {query}

# And the answer:
# {response}

# Determine whether the answer is fully grounded in the provided context. 
# Respond with "Yes" or "No" and provide a short reason.
# """
#     output = query_ollama(prompt)
#     return "yes" in output.lower()

# def answer_relevance_ollama(query, response):
#     prompt = f"""
#     Question: {query}

#     Answer: {response}

#     Is the answer relevant to the question? Respond with "Yes" or "No" and explain.
#     """
#     output = query_ollama(prompt)
#     return "yes" in output.lower()

# def context_relevance_ollama(query, context_chunk):
#     prompt = f"""
#     Question: {query}

#     Context: {context_chunk}

#     Is this context relevant to answering the question? Reply "Yes" or "No" and why.
#     """
#     output = query_ollama(prompt)
#     return "yes" in output.lower()


# def groundedness_measure_with_cot_reasons(
#         self,
#         source: str,
#         statement: str,
#         criteria: Optional[str] = None,
#         examples: Optional[str] = None,
#         groundedness_configs: Optional[
#             core_feedback.GroundednessConfigs
#         ] = None,
#         min_score_val: int = 0,
#         max_score_val: int = 3,
#         temperature: float = 0.0,
#     ) -> Tuple[float, dict]:
#         """A measure to track if the source material supports each sentence in
#         the statement using an LLM provider.

#         The statement will first be split by a tokenizer into its component sentences.

#         Then, trivial statements are eliminated so as to not dilute the evaluation. Note that if all statements are filtered out as trivial, returns 0.0 with a reason indicating no non-trivial statements were found.

#         The LLM will process each statement, using chain of thought methodology to emit the reasons.

#         Abstentions will be considered as grounded.

#         Example:
#             ```python
#             from trulens.core import Feedback
#             from trulens.providers.openai import OpenAI

#             provider = OpenAI()

#             f_groundedness = (
#                 Feedback(provider.groundedness_measure_with_cot_reasons)
#                 .on(context.collect())
#                 .on_output()
#                 )
#             ```

#         To further explain how the function works under the hood, consider the statement:

#         "Hi. I'm here to help. The university of Washington is a public research university. UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

#         The function will split the statement into its component sentences:

#         1. "Hi."
#         2. "I'm here to help."
#         3. "The university of Washington is a public research university."
#         4. "UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

#         Next, trivial statements are removed, leaving only:

#         3. "The university of Washington is a public research university."
#         4. "UW's connections to major corporations in Seattle contribute to its reputation as a hub for innovation and technology"

#         The LLM will then process the statement, to assess the groundedness of the statement.

#         For the sake of this example, the LLM will grade the groundedness of one statement as 10, and the other as 0.

#         Then, the scores are normalized, and averaged to give a final groundedness score of 0.5.

#         Args:
#             source (str): The source that should support the statement.
#             statement (str): The statement to check groundedness.
#             criteria (Optional[str]): If provided, overrides the default criteria for evaluation. Defaults to None.
#             examples (Optional[str]): Optional examples to guide the evaluation. Defaults to None.
#             groundedness_configs (Optional[core_feedback.GroundednessConfigs]): Configuration for groundedness evaluation. Defaults to None.
#             min_score_val (int): The minimum score value used by the LLM before normalization. Defaults to 0.
#             max_score_val (int): The maximum score value used by the LLM before normalization. Defaults to 3.
#             temperature (float): The temperature for the LLM response, which might have impact on the confidence level of the evaluation. Defaults to 0.0.

#         Returns:
#             Tuple[float, dict]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a dictionary containing the reasons for the evaluation.
#         """

#         assert self.endpoint is not None, "Endpoint is not set."

#         groundedness_scores = {}
#         reasons_list = []

#         use_sent_tokenize = (
#             groundedness_configs.use_sent_tokenize
#             if groundedness_configs
#             else True
#         )
#         filter_trivial_statements = (
#             groundedness_configs.filter_trivial_statements
#             if groundedness_configs
#             else True
#         )

#         if use_sent_tokenize:
#             nltk.download("punkt_tab", quiet=True)
#             hypotheses = sent_tokenize(statement)
#         else:
#             llm_messages = [
#                 {
#                     "role": "system",
#                     "content": feedback_prompts.LLM_GROUNDEDNESS_SENTENCES_SPLITTER,
#                 },
#                 {"role": "user", "content": statement},
#             ]

#             hypotheses = self.endpoint.run_in_pace(
#                 func=self._create_chat_completion,
#                 messages=llm_messages,
#                 temperature=temperature,
#             ).split("\n")

#         # Remove simple numeric list markers such as "1." or "2)"
#         hypotheses = [
#             h for h in hypotheses if not re.match(r"^\s*\d+[\.)]?\s*$", h)
#         ]

#         if filter_trivial_statements:
#             hypotheses = self._remove_trivial_statements(hypotheses)

#             if not hypotheses:
#                 return 0.0, {"reason": "No non-trivial statements to evaluate"}

#         output_space = self._determine_output_space(
#             min_score_val, max_score_val
#         )

#         system_prompt = feedback_v2.Groundedness.generate_system_prompt(
#             min_score=min_score_val,
#             max_score=max_score_val,
#             criteria=criteria,
#             examples=examples,
#             output_space=output_space,
#         )

#         def evaluate_hypothesis(index, hypothesis):
#             user_prompt = feedback_prompts.LLM_GROUNDEDNESS_USER.format(
#                 premise=f"{source}", hypothesis=f"{hypothesis}"
#             )
#             score, reason = self.generate_score_and_reasons(
#                 system_prompt=system_prompt,
#                 user_prompt=user_prompt,
#                 min_score_val=min_score_val,
#                 max_score_val=max_score_val,
#                 temperature=temperature,
#             )

#             # Build structured reason dict to ensure criteria, evidence, and score present
#             structured = {
#                 "criteria": hypothesis,
#                 "supporting_evidence": reason.get("reason", ""),
#                 "score": score,
#             }
#             return index, score, structured

#         results = []

#         with ThreadPoolExecutor() as executor:
#             futures = [
#                 executor.submit(evaluate_hypothesis, i, hypothesis)
#                 for i, hypothesis in enumerate(hypotheses)
#             ]

#             for future in as_completed(futures):
#                 results.append(future.result())

#         results.sort(key=lambda x: x[0])  # Sort results by index

#         for i, score, reason in results:
#             groundedness_scores[f"statement_{i}"] = score
#             reasons_list.append(reason)

#         # Calculate the average groundedness score from the scores dictionary
#         average_groundedness_score = float(
#             np.mean(list(groundedness_scores.values()))
#         )

#         return average_groundedness_score, {"reasons": reasons_list}