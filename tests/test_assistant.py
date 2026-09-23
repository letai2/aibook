"""Offline contracts: proposed data never gets arbitrary execution authority."""
import json
import unittest

import torch

from mini_gpt.assistant import (BackendError, MiniGPTBackend, PROTOCOL,
                                ScriptedFixture, run_assistant)
from mini_gpt.config import ModelConfig
from mini_gpt.context import ContextItem, build_context
from mini_gpt.memory import MemoryRecord, MemoryStore
from mini_gpt.model import MiniGPT
from mini_gpt.reasoning import (execute_plan, majority_answer, select_verified,
                                study_minutes, verify_study_answer)
from mini_gpt.retrieval import COURSE_DOCUMENTS, chunk_document, keyword_search
from mini_gpt.tokenizer import CharacterTokenizer
from mini_gpt.tools import ToolError, execute_tool, strict_object, validate_call


CALL = {"action":"tool", "name":"multiply", "arguments":{"a":25,"b":48}}
ANSWER = {"action":"finish", "answer":"1200", "citations":[]}


class ToolTests(unittest.TestCase):
    def test_operations_and_zero(self):
        for name, expected in (("add",73),("subtract",-23),("multiply",1200)):
            result = execute_tool({"name":name,"arguments":{"a":25,"b":48}})
            self.assertTrue(result.ok)
            self.assertEqual(result.value,expected)
        self.assertEqual(execute_tool({"name":"multiply","arguments":{"a":0,"b":8}}).value,0)

    def test_argument_validation(self):
        for b in (True,False,2.,"2",None,[],1000001,-1000001):
            result = execute_tool({"name":"add","arguments":{"a":1,"b":b}})
            self.assertFalse(result.ok)
            self.assertIsNone(result.value)
        for payload in ({},[], {"name":"eval","arguments":{"a":1,"b":2}},
                        {"name":"add","arguments":{"a":1,"b":2},"permission":True},
                        {"name":"add","arguments":{"a":1,"b":2,"c":3}}):
            with self.assertRaises(ToolError): validate_call(payload)

    def test_json_is_not_python_and_duplicate_keys_fail(self):
        for text in ('__import__("os")', '{"a":1,"a":2}', '{"a":NaN}', '[]', '', 'x'*4097):
            with self.assertRaises(ToolError): strict_object(text)
        call = validate_call('{"name":"add","arguments":{"a":2,"b":3}}')
        self.assertEqual(call.payload(),{"name":"add","arguments":{"a":2,"b":3}})


class ReasoningTests(unittest.TestCase):
    def test_dependent_plan_and_independent_verification(self):
        correct = [
            {"id":"daily","name":"add","arguments":{"a":25,"b":20}},
            {"id":"total","name":"multiply","arguments":{"a":"daily","b":3}},
        ]
        trace = execute_plan(correct)
        self.assertEqual((trace[0].value,trace[1].a,trace[1].value),(45,45,135))
        wrong = [
            {"id":"labs","name":"multiply","arguments":{"a":20,"b":3}},
            {"id":"total","name":"add","arguments":{"a":25,"b":"labs"}},
        ]
        self.assertFalse(verify_study_answer([25,20],3,execute_plan(wrong)[-1].value))
        self.assertEqual(study_minutes([10,5],4),60)
        self.assertFalse(verify_study_answer([1],1,True))

    def test_invalid_plans_stop_before_execution(self):
        valid = {"id":"one","name":"add","arguments":{"a":1,"b":2}}
        for plan in ([],[valid,valid],
                     [{"id":"one","name":"add","arguments":{"a":"future","b":2}}],
                     [{"id":"one","name":"add","arguments":{"a":True,"b":2}}]):
            with self.assertRaises(ToolError): execute_plan(plan)
        with self.assertRaises(ToolError): execute_plan([valid,valid],max_steps=1)

    def test_vote_is_not_truth(self):
        answers = [85,85,135]
        self.assertEqual(majority_answer(answers).answer,85)
        self.assertEqual(select_verified(answers,lambda a: verify_study_answer([25,20],3,a)),135)
        self.assertTrue(majority_answer([1,2]).tied)
        self.assertIsNone(majority_answer([]).answer)
        self.assertIsNone(select_verified([1,2],lambda a: a > 0))
        self.assertEqual(select_verified([0,0],lambda a: a == 0),0)


class AssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bounded_tool_roundtrip(self):
        fixture = ScriptedFixture([CALL,ANSWER])
        result = run_assistant("25 * 48?",fixture,
            verify_answer=lambda a,c,t: a == "1200" and t[0]["value"] == 1200)
        self.assertEqual(result.status,"finished")
        self.assertTrue(result.verified)
        self.assertEqual(fixture.calls,2)
        self.assertEqual(result.tool_results[0]["arguments"],{"a":25,"b":48})
        self.assertIn("NOT model capability",result.backend)
        self.assertIn("tool:0",result.context_ids)

    def test_unchecked_completion_is_not_verification(self):
        result = run_assistant("answer?",ScriptedFixture([ANSWER]))
        self.assertEqual(result.status,"finished")
        self.assertFalse(result.verified)
        bad = run_assistant("answer?",ScriptedFixture([ANSWER]),verify_answer=lambda a,c,t: False)
        self.assertEqual(bad.status,"verification_failed")

    def test_budget_repetition_and_malformed_output(self):
        cases = [([CALL],{"max_tool_calls":0},"tool_limit",0),
                 ([CALL],{"max_steps":1},"step_limit",1),
                 ([CALL,CALL],{},"repeated_action",1),
                 (["not JSON"],{},"invalid_action",0),
                 ([{"action":"tool","name":"shell","arguments":{"a":1,"b":2}}],{},"invalid_action",0)]
        for responses,kwargs,status,count in cases:
            result = run_assistant("25 * 48?",ScriptedFixture(responses),**kwargs)
            self.assertEqual(result.status,status)
            self.assertEqual(len(result.tool_results),count)

    def test_context_cannot_silently_drop_controller(self):
        fixture = ScriptedFixture([ANSWER],context_window=100,max_new_tokens=10)
        result = run_assistant("question",fixture)
        self.assertEqual(result.status,"context_limit")
        self.assertEqual(fixture.calls,0)

    def test_evidence_and_memory_are_observable_not_automatic_truth(self):
        chunks = [chunk for document in COURSE_DOCUMENTS for chunk in chunk_document(document)]
        evidence = keyword_search("Checkpoint",chunks,k=1)[0].chunk
        memory = MemoryStore([MemoryRecord("unit","minutes","explicit user statement")])
        original = memory.records()
        response = {"action":"finish","answer":evidence.text,"citations":[evidence.id]}
        result = run_assistant("Checkpoint",ScriptedFixture([response]),chunks=chunks,
                               memory=memory,memory_keys=["unit"])
        self.assertEqual(result.status,"finished")
        self.assertFalse(result.verified)
        self.assertIn(evidence.id,result.context_ids)
        self.assertIn(evidence.id,next(event["ids"] for event in result.events if event["state"] == "retrieve"))
        self.assertIn("memory:unit",result.context_ids)
        self.assertEqual(memory.records(),original)
        response["citations"] = ["invented-source"]
        self.assertEqual(run_assistant("Checkpoint",ScriptedFixture([response]),chunks=chunks).status,"invalid_action")

    def test_memory_requires_caller_selected_keys_and_never_writes(self):
        class RecordingFixture(ScriptedFixture):
            def complete(self, prompt):
                self.prompt = prompt
                return super().complete(prompt)

        memory = MemoryStore([MemoryRecord("unit","minutes","explicit unit preference"),
                              MemoryRecord("private","not authorized here","different task")])
        original = memory.records()
        for keys,expected in (((),()),(["missing","unit"],("memory:unit",))):
            fixture = RecordingFixture([ANSWER])
            result = run_assistant("answer?",fixture,memory=memory,memory_keys=keys)
            self.assertEqual(tuple(key for key in result.context_ids if key.startswith("memory:")),expected)
            self.assertNotIn("not authorized here",fixture.prompt)
            self.assertNotIn("memory:private",fixture.prompt)
            self.assertEqual(memory.records(),original)
        omitted = run_assistant("answer?",RecordingFixture([ANSWER]),memory=memory)
        self.assertFalse(any(key.startswith("memory:") for key in omitted.context_ids))
        for keys in (None,"unit",["unit","unit"],[""],[1]):
            with self.assertRaises(ValueError):
                run_assistant("answer?",ScriptedFixture([ANSWER]),memory=memory,memory_keys=keys)

    def test_fixture_enforces_declared_output_budget(self):
        fixture = ScriptedFixture(["longer"],max_new_tokens=2)
        with self.assertRaises(BackendError): fixture.complete("question")
        self.assertEqual(fixture.calls,1)
        self.assertEqual(ScriptedFixture(["ok"],max_new_tokens=2).complete("question"),"ok")
        for budget in (0,True,4096):
            with self.assertRaises(ValueError): ScriptedFixture([],max_new_tokens=budget)

    def test_fixture_is_exhaustible_not_a_fallback_model(self):
        result = run_assistant("answer?",ScriptedFixture([]))
        self.assertEqual(result.status,"backend_error")
        self.assertEqual(result.answer,"")

    def test_real_minigpt_runs_without_updating_weights(self):
        torch.manual_seed(23)
        prompt = "25*48="
        tokenizer = CharacterTokenizer.from_text(prompt)
        model = MiniGPT(ModelConfig(tokenizer.vocab_size,32,8,2,1,0.)).train()
        before = [p.detach().clone() for p in model.parameters()]
        backend = MiniGPTBackend(model,tokenizer,max_new_tokens=4)
        text = backend.complete(prompt)
        self.assertEqual(len(text),4)
        self.assertEqual(backend.last_counts,{"prompt_tokens":6,"generated_tokens":4})
        self.assertEqual(backend.calls,1)
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,model.parameters())))
        with self.assertRaises(BackendError): backend.complete("unknown character")
        with self.assertRaises(BackendError): backend.complete(prompt*6)

    def test_real_model_format_failure_is_not_replaced_by_fixture(self):
        torch.manual_seed(23)
        question = "25*48?"
        pack = build_context(question,[ContextItem("controller:protocol",PROTOCOL,"instruction")],
                             max_tokens=4096,reserve_tokens=4,count_tokens=len)
        tokenizer = CharacterTokenizer.from_text(pack.prompt)
        model = MiniGPT(ModelConfig(tokenizer.vocab_size,len(pack.prompt)+8,8,2,1,0.))
        backend = MiniGPTBackend(model,tokenizer,max_new_tokens=4)
        result = run_assistant(question,backend)
        self.assertEqual(backend.calls,1)
        self.assertEqual(result.status,"invalid_action")
        self.assertEqual(result.answer,"")
        self.assertEqual(len(result.tool_results),0)


if __name__ == "__main__":
    unittest.main()
