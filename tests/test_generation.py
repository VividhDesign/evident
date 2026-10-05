import json

from evident.judge import Judge
from evident.llm import FakeLLM, extract_json
from evident.rag import ABSTAIN_TEXT, RAG, is_abstention, normalize_citations, sentence_citations


def test_citation_parsing():
    cites = sentence_citations("Roth IRAs use after-tax money [1]. Withdrawals are tax free [1][2]. Nice.")
    assert cites == [("Roth IRAs use after-tax money .", [1]), ("Withdrawals are tax free .", [1, 2]), ("Nice.", [])]
    # markers written after the full stop belong to the sentence before them
    after = sentence_citations("Bonds pay coupons. [2] Prices fall when rates rise. [3][4]")
    assert after == [("Bonds pay coupons.", [2]), ("Prices fall when rates rise.", [3, 4])]


def test_abstention_detection():
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("I do not have enough information to answer.")
    assert not is_abstention("Roth IRAs are funded with after-tax dollars [1].")


def test_extract_json_tolerates_wrapping():
    assert extract_json('```json\n{"score": 4}\n```') == {"score": 4}
    assert extract_json("no json here") is None


def test_rag_answer_with_citations_and_cost(retriever):
    llm = FakeLLM(["A Roth IRA is funded with after-tax dollars [1]. Withdrawals are tax free [1][9]."])
    ans = RAG(retriever, llm, k=3).answer("How is a Roth IRA taxed?")
    assert not ans.abstained
    assert len(ans.sources) == 3 and ans.sources[0].number == 1
    assert ans.invalid_citations == 1  # [9] does not exist
    assert ans.citation_coverage == 1.0
    prompt = llm.calls[0][1]["content"]
    assert "[1]" in prompt and "Question: How is a Roth IRA taxed?" in prompt
    assert ans.input_tokens == 100 and ans.cost_usd == 0.0


def test_judge_scores(retriever):
    gen = FakeLLM(["Roth IRA money is after-tax [1]. It is tax free later [2]."])
    ans = RAG(retriever, gen, k=2).answer("roth ira")

    def judge_reply(messages):
        prompt = messages[0]["content"]
        if "factual claims" in prompt:
            return json.dumps({"claims": [{"claim": "after-tax", "supported": True},
                                          {"claim": "tax free later", "supported": False}]})
        if "How directly" in prompt:
            return '{"score": 4, "reason": "ok"}'
        return '{"verdicts": [{"pair": 1, "supported": true}, {"pair": 2, "supported": false}]}'

    j = Judge(FakeLLM(judge_reply)).judge(ans)
    assert j.faithfulness == 0.5 and j.relevance == 4.0
    assert j.citation_precision == 0.5 and j.citation_pairs == 2


def test_judge_skips_abstentions(retriever):
    ans = RAG(retriever, FakeLLM([ABSTAIN_TEXT]), k=2).answer("what is the capital of mars")
    assert ans.abstained
    j = Judge(FakeLLM([])).judge(ans)
    assert j.faithfulness is None and j.relevance is None


def test_normalize_citations():
    text = normalize_citations("Pay the debt first 【1】 【4】. Invest the rest【2†L3-L5】. Balance risk ［3］.")
    assert text == "Pay the debt first [1] [4]. Invest the rest [2]. Balance risk [3]."
    assert [c for _, c in sentence_citations(text)] == [[1, 4], [2], [3]]
