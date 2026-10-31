"""Offline evaluation harness for the context middleware (no LLM calls).

Uses the NaturalQuestions multi-document QA data from "Lost in the Middle"
(Liu et al.): 2,655 questions, each with 20 Wikipedia passages of which exactly
one ("gold") contains the answer and 19 are retrieved distractors.
"""
