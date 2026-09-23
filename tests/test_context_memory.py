"""Boundaries of the offline educational context, retrieval and memory pieces."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mini_gpt.context import ContextItem, build_context
from mini_gpt.memory import Message, MemoryRecord, MemoryStore, format_history, summarize_history
from mini_gpt.retrieval import (Document, Chunk, SearchHit, VectorIndex, answer_from_hits,
                                chunk_document, cosine_similarity, keyword_search, terms)


class ContextTests(unittest.TestCase):
    def test_full_serialization_including_question_and_reserve_is_counted(self):
        item = ContextItem("source", 'Say "yes".', "evidence")
        full = build_context("Why?", [item], max_tokens=1000, reserve_tokens=20, count_tokens=len)
        self.assertEqual(full.token_count, len(full.prompt))
        self.assertEqual(json.loads(full.prompt)["context"][0]["text"], item.text)
        exact = build_context("Why?", [item], max_tokens=full.token_count+20,
                              reserve_tokens=20, count_tokens=len)
        self.assertEqual(exact.included_ids, ("source",))
        short = build_context("Why?", [item], max_tokens=full.token_count+19,
                              reserve_tokens=20, count_tokens=len)
        self.assertEqual(short.omitted_ids, ("source",))
        self.assertEqual(json.loads(short.prompt)["question"], "Why?")

    def test_oversized_item_does_not_block_later_small_item(self):
        items = [ContextItem("large", "a"*300), ContextItem("small", "yes")]
        pack = build_context("Q", items, max_tokens=100, reserve_tokens=10, count_tokens=len)
        self.assertEqual(pack.included_ids, ("small",))
        self.assertEqual(pack.omitted_ids, ("large",))
        self.assertLessEqual(pack.token_count+10, 100)

    def test_counter_receives_complete_candidate_not_fragments(self):
        seen = []

        def counter(text):
            payload = json.loads(text)
            self.assertEqual(set(payload), {"question", "context"})
            seen.append(text)
            return len(text.split())  # Deliberately different from len.

        pack = build_context("one two", [ContextItem("s", "three four")],
                             max_tokens=100, reserve_tokens=10, count_tokens=counter)
        self.assertTrue(seen)
        self.assertEqual(pack.token_count, len(pack.prompt.split()))

    def test_invalid_budget_duplicates_and_missing_question(self):
        for maximum, reserved in [(0,0), (10,10), (10,-1), (True,0), (10,False)]:
            with self.subTest(maximum=maximum,reserved=reserved), self.assertRaises(ValueError):
                build_context("Q", [], max_tokens=maximum, reserve_tokens=reserved, count_tokens=len)
        for question, items, maximum in [("",[],100), ("Q",[ContextItem("s","a")]*2,100), ("Q",[],1)]:
            with self.assertRaises(ValueError):
                build_context(question,items,max_tokens=maximum,reserve_tokens=0,count_tokens=len)
        with self.assertRaises(ValueError):
            build_context("Q",[],max_tokens=100,reserve_tokens=0,count_tokens=lambda _: True)


class RetrievalTests(unittest.TestCase):
    def test_chunk_boundaries_overlap_and_exact_offsets(self):
        document = Document("guide", "  alpha\tb c d e f g  ")
        chunks = chunk_document(document,chunk_words=4,overlap_words=1)
        self.assertEqual([chunk.text.split() for chunk in chunks],
                         [["alpha","b","c","d"],["d","e","f","g"]])
        self.assertTrue(all(document.text[c.start:c.end] == c.text for c in chunks))
        self.assertEqual(chunks[0].start,2)
        self.assertEqual(len({c.id for c in chunks}),len(chunks))
        self.assertEqual(chunk_document(Document("empty"," \t"),chunk_words=4,overlap_words=1),[])

    def test_chunk_unicode_offsets_count_codepoints_not_utf8_bytes(self):
        document = Document("unicode", "  الف\tب  ")
        chunks = chunk_document(document, chunk_words=1, overlap_words=0)
        self.assertEqual([(c.start, c.end, c.text) for c in chunks],
                         [(2, 5, "الف"), (6, 7, "ب")])
        self.assertTrue(all(document.text[c.start:c.end] == c.text for c in chunks))
        self.assertNotEqual(len(document.text[:chunks[0].end].encode('utf-8')), chunks[0].end)

    def test_chunk_last_short_piece_and_invalid_stride(self):
        chunks = chunk_document(Document("x","a b c d e"),chunk_words=3,overlap_words=0)
        self.assertEqual([c.text for c in chunks],["a b c","d e"])
        for size, overlap in [(0,0),(3,3),(3,-1),(3,True),(True,0)]:
            with self.assertRaises(ValueError):
                chunk_document(Document("x","a b"),chunk_words=size,overlap_words=overlap)

    def test_search_terms_are_not_character_tokenizer_or_stemming(self):
        self.assertEqual(terms("CPU، می‌روم!"),["cpu","می‌روم"])
        self.assertNotEqual(terms("ی"),terms("ي"))
        self.assertEqual(terms("  ... "),[])

    def test_keyword_order_ties_empty_and_no_overlap(self):
        chunks = [chunk_document(Document(i,text),chunk_words=4,overlap_words=0)[0]
                  for i,text in [("b","CPU book"),("a","CPU class"),("c","GPU")]]
        self.assertEqual([h.chunk.document_id for h in keyword_search("CPU",chunks)],["a","b"])
        self.assertEqual(keyword_search("unknown",chunks),[])
        self.assertEqual(keyword_search("",chunks),[])
        self.assertEqual(len(keyword_search("CPU",chunks,k=1)),1)
        with self.assertRaises(ValueError):
            keyword_search("CPU",chunks,k=0)

    def test_cosine_known_values_and_zero_policy(self):
        self.assertAlmostEqual(cosine_similarity([1,0],[2,0]),1)
        self.assertAlmostEqual(cosine_similarity([1,0],[1,1]),1/math.sqrt(2))
        self.assertAlmostEqual(cosine_similarity([1,0],[-1,0]),-1)
        self.assertEqual(cosine_similarity([0,0],[1,2]),0)
        self.assertEqual(cosine_similarity([],[]),0)
        with self.assertRaises(ValueError):
            cosine_similarity([1],[1,2])
        with self.assertRaises(ValueError):
            cosine_similarity([float("nan")],[1])

    def test_count_vector_index_has_shared_axes_and_no_synonym_claim(self):
        chunks = [chunk_document(Document(i,text),chunk_words=4,overlap_words=0)[0]
                  for i,text in [("a","save save"),("b","record save")]]
        index = VectorIndex(chunks)
        self.assertEqual(index.vocabulary,tuple(sorted(["record","save"])))
        self.assertEqual([h.chunk.document_id for h in index.search("save")],["a","b"])
        self.assertEqual([h.chunk.document_id for h in index.search("save",min_score=0.9)],["a"])
        self.assertEqual(index.search("keep"),[])
        self.assertEqual(VectorIndex([]).search("any"),[])
        with self.assertRaises(ValueError):
            index.search("save",min_score=float("nan"))
        with self.assertRaises(ValueError):
            VectorIndex(chunks+chunks)

    def test_cosine_finite_extreme_values_keep_direction_and_range(self):
        for magnitude in (1e308, 1e-308):
            with self.subTest(magnitude=magnitude):
                self.assertAlmostEqual(cosine_similarity([magnitude,0],[magnitude,0]),1)
                self.assertAlmostEqual(cosine_similarity([magnitude,magnitude],
                                                         [-magnitude,-magnitude]),-1)
                score = cosine_similarity([magnitude,magnitude],[magnitude,0])
                self.assertTrue(math.isfinite(score))
                self.assertAlmostEqual(score,1/math.sqrt(2))

    def test_extractive_answer_preserves_conflict_without_claiming_a_resolution(self):
        chunks = [Chunk("a","one","Use CPU.",0,8), Chunk("b","two","Use GPU.",0,8)]
        answer = answer_from_hits([SearchHit(c,1.) for c in chunks])
        self.assertEqual(answer.citations,("a","b"))
        self.assertIn("[a] Use CPU.",answer.text)
        self.assertIn("[b] Use GPU.",answer.text)
        self.assertFalse(answer.abstained)
        self.assertTrue(answer_from_hits([]).abstained)
        self.assertTrue(answer_from_hits([SearchHit(chunks[0],0.)]).abstained)
        self.assertTrue(answer_from_hits([SearchHit(chunks[0],0.2)],min_score=0.3).abstained)
        with self.assertRaises(ValueError):
            answer_from_hits([SearchHit(chunks[0],1.)]*2)

    def test_actual_toy_embedding_training_is_separate_from_the_lexical_index(self):
        from mini_gpt.retrieval import train_tiny_embeddings
        names,before,after,initial,final = train_tiny_embeddings()
        self.assertEqual(len(names),4)
        self.assertEqual(len(after[0]),2)
        self.assertLess(final,initial)
        self.assertNotEqual(before,after)
        self.assertGreater(cosine_similarity(after[0],after[1]),0.9)
        self.assertLess(cosine_similarity(after[0],after[2]),-0.9)
        _,zero_before,zero_after,first,last = train_tiny_embeddings(steps=0)
        self.assertEqual(zero_before,zero_after)
        self.assertEqual(first,last)


class MemoryTests(unittest.TestCase):
    def test_history_keeps_roles_and_does_not_invent_model_state(self):
        messages = [Message("user",'Only "CPU".\nNo install.'),Message("assistant","OK.")]
        payload = json.loads(format_history(messages))
        self.assertEqual(payload[0],{"role":"user","content":messages[0].content})
        self.assertEqual(len(messages),2)
        self.assertEqual(format_history([]),"[]")
        with self.assertRaises(ValueError):
            Message("system","not a supported history role here")
        with self.assertRaises(ValueError):
            Message("user"," ")

    def test_summary_is_explicitly_lossy_even_with_provenance(self):
        messages = [Message("user","Run the exercise.\nCPU only."),Message("assistant","Open it.")]
        summary = summarize_history(messages,max_chars=200)
        self.assertEqual(summary.source_turns,(1,2))
        self.assertEqual(summary.omitted_turns,())
        self.assertNotIn("CPU",summary.text)
        empty = summarize_history(messages,max_chars=0)
        self.assertEqual((empty.text,empty.source_turns,empty.omitted_turns),("",(),(1,2)))
        self.assertEqual(len(messages),2)

    def test_summary_skips_long_line_and_can_keep_a_later_short_one(self):
        messages = [Message("user","x"*100),Message("user","CPU")]
        summary = summarize_history(messages,max_chars=15)
        self.assertEqual(summary.source_turns,(2,))
        self.assertEqual(summary.omitted_turns,(1,))
        self.assertEqual(summary.text,"2:user: CPU")

    def test_store_updates_preserves_source_and_forgets(self):
        store = MemoryStore([MemoryRecord("device","CPU","user-1")])
        store.upsert(MemoryRecord("device","GPU","user-3"))
        self.assertEqual(store.records(),(MemoryRecord("device","GPU","user-3"),))
        self.assertTrue(store.forget("device"))
        self.assertFalse(store.forget("device"))
        self.assertEqual(store.records(),())
        with self.assertRaises(ValueError):
            MemoryStore([MemoryRecord("a","b","c")]*2)

    def test_persistence_changes_only_at_explicit_save(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"memory.json"
            store = MemoryStore([MemoryRecord("device","CPU","user-1")])
            self.assertFalse(path.exists())
            store.save(path)
            store.upsert(MemoryRecord("device","GPU","user-3"))
            self.assertEqual(MemoryStore.load(path).records()[0].value,"CPU")
            store.save(path)
            self.assertEqual(MemoryStore.load(path).records()[0].source,"user-3")
            store.forget("device")
            self.assertEqual(len(MemoryStore.load(path).records()),1)
            store.save(path)
            self.assertEqual(MemoryStore.load(path).records(),())

    def test_memory_file_format_and_record_provenance_are_validated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"memory.json"
            for value in [{"version":True,"records":[]}, {"version":2,"records":[]},
                          {"version":1,"records":[{"key":"device","value":"CPU"}]},
                          {"version":1,"records":[{"key":"device","value":"CPU","source":""}]}]:
                path.write_text(json.dumps(value),encoding="utf-8")
                with self.assertRaises(ValueError):
                    MemoryStore.load(path)


if __name__ == "__main__":
    unittest.main()
