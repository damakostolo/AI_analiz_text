## ai_core.py
from __future__ import annotations
import json
from typing import Dict, Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Doc

import pandas as pd
from transformers import pipeline

from models import ActionTriple
from visualization import Visualizer


class LiteraryAI:
    def __init__(self, prefer_trf: bool = True, gpu: bool = False, max_doc_len: int | None = None,
                 visualizer: Visualizer | None = None):
        self.nlp = self._load_spacy(prefer_trf)
        self.emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0 if gpu else -1,
            truncation=True
        )
        self.max_doc_len = max_doc_len
        self.visualizer = visualizer or Visualizer()

    def _load_spacy(self, prefer_trf: bool) -> Language:
        tried = []
        if prefer_trf:
            try:
                return spacy.load("en_core_web_trf")
            except Exception as e:
                tried.append(("en_core_web_trf", str(e)))
        try:
            return spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(f"Cannot load spaCy models. Tried: {tried + [('en_core_web_sm', str(e))]}")


    def read_text(self, text: str) -> Doc:
        if self.max_doc_len:
            text = text[: self.max_doc_len]
        return self.nlp(text)


    def extract_characters(self, doc: Doc, top_k: int = 50) -> pd.DataFrame:
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if not persons:
            return pd.DataFrame(columns=["character", "count"])
        s = pd.Series(persons).str.strip().str.replace(r"\s+", " ", regex=True)
        # Просте зведення варіантів імен (John, Mr. John → John)
        s = s.str.replace(r"^(Mr\.|Mrs\.|Ms\.|Dr\.|Sir)\s+", "", regex=True)
        counts = s.value_counts().head(top_k)
        return counts.rename_axis("character").reset_index(name="count")


    def extract_svo(self, doc: Doc) -> List[ActionTriple]:
        triples: List[ActionTriple] = []
        for i, sent in enumerate(doc.sents):
            root = None
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root = token
                    break
            if not root:
                continue
            subj = [w for w in root.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            dobj = [w for w in root.rights if w.dep_ in ("dobj", "pobj", "attr", "dative")]
            if subj and dobj:
                triples.append(ActionTriple(
                    subject=subj[0].text,
                    verb=root.lemma_.lower(),
                    object=dobj[0].text,
                    sent_id=i
                ))
        return triples

  

    def emotion_curve(self, doc: Doc, step: int = 5) -> pd.DataFrame:
        """Аналіз емоцій всього тексту → середній тон по кожній категорії"""
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sents:
            return pd.DataFrame(columns=["emotion", "mean_tone"])

        preds = self.emotion(sents, batch_size=16, truncation=True)

        # берем максимальную эмоцию для каждого предложения
        primary = [max(p, key=lambda x: x["score"]) for p in preds]

        tone_map = {
            "joy": 1.0,
            "surprise": 0.6,
            "neutral": 0.0,
            "sadness": -0.8,
            "anger": -0.7,
            "disgust": -0.9,
            "fear": -1.0,
        }

        df = pd.DataFrame({
            "emotion": [p["label"] for p in primary],
            "tone": [tone_map.get(p["label"], 0.0) for p in primary],
        })

        # считаем среднее значение по каждой эмоции
        agg = df.groupby("emotion", as_index=False)["tone"].mean()

        # сортируем эмоции в логическом порядке
        order = ["joy", "surprise", "neutral", "sadness", "anger", "disgust", "fear"]
        agg["emotion"] = pd.Categorical(agg["emotion"], categories=order, ordered=True)
        agg = agg.sort_values("emotion").reset_index(drop=True)

        return agg



  
    # ---------- End-to-end ----------
    def analyze(self, text: str, out_dir: str) -> Dict[str, str]:
        doc = self.read_text(text)
        # Characters
        char_df = self.extract_characters(doc)
        char_csv = f"{out_dir}/characters.csv"
        char_df.to_csv(char_csv, index=False)
        self.visualizer.plot_top_characters(char_df, f"{out_dir}/characters.png")
        # SVO
        triples = self.extract_svo(doc)
        svo_csv = f"{out_dir}/actions.csv"
        pd.DataFrame([t.__dict__ for t in triples]).to_csv(svo_csv, index=False)
        G = self.visualizer.build_action_graph(triples)
        self.visualizer.plot_action_graph(G, f"{out_dir}/actions_graph.png")
        # Emotions
        curve = self.emotion_curve(doc)
        emo_csv = f"{out_dir}/emotion_curve.csv"
        curve.to_csv(emo_csv, index=False)
        self.visualizer.plot_emotion_curve(curve, f"{out_dir}/emotion_curve.png")
        # Summary JSON
        summary = {
            "characters_csv": char_csv,
            "characters_png": f"{out_dir}/characters.png",
            "actions_csv": svo_csv,
            "actions_graph_png": f"{out_dir}/actions_graph.png",
            "emotion_curve_csv": emo_csv,
            "emotion_curve_png": f"{out_dir}/emotion_curve.png",
        }
        with open(f"{out_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary