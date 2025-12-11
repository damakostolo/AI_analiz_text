## ai_core.py
from __future__ import annotations
import re
import math
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

import spacy
from spacy.language import Language
from spacy.tokens import Doc

import pandas as pd
import numpy as np
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class ActionTriple:
    subject: str
    verb: str
    object: str
    sent_id: int


class LiteraryAI:
    def __init__(self, prefer_trf: bool = True, gpu: bool = False, max_doc_len: int | None = None):
        self.nlp = self._load_spacy(prefer_trf)
        self.emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=0 if gpu else -1,
            truncation=True
        )
        self.max_doc_len = max_doc_len

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



  
    # ---------- Graph build & save ----------
    def build_action_graph(self, triples: Iterable[ActionTriple]) -> nx.DiGraph:
        G = nx.DiGraph()
        for t in triples:
            s = t.subject.strip()
            o = t.object.strip()
            v = t.verb
            if s and o:
                G.add_edge(s, o, label=v, sent_id=t.sent_id)
        return G

    # ---------- Plotting helpers ----------

    def build_action_graph(self, triples: Iterable[ActionTriple], min_count: int = 2) -> nx.DiGraph:
        # Считаем частоту появления субъекта и объекта
        subjects = {}
        objects = {}
        for t in triples:
            subjects[t.subject] = subjects.get(t.subject, 0) + 1
            objects[t.object] = objects.get(t.object, 0) + 1

        # Строим граф только из частых элементов
        G = nx.DiGraph()
        for t in triples:
            if subjects[t.subject] < min_count or objects[t.object] < min_count:
                continue
            s, o, v = t.subject.strip(), t.object.strip(), t.verb
            if s and o:
                G.add_edge(s, o, label=v, sent_id=t.sent_id)
        return G


    def plot_emotion_curve(self, curve_df: pd.DataFrame, out_path: str):
        """Строим итоговую кривую по 7 эмоциям"""
        if curve_df.empty:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(curve_df["emotion"], curve_df["tone"],
                 color="purple", marker="o", linewidth=2)
        plt.ylim(-1.05, 1.05)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title("Overall Emotional Tone")
        plt.ylabel("Tone (-1 negative, +1 positive)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    def plot_top_characters(self, char_df: pd.DataFrame, out_path: str, top_n: int = 20):
        if char_df.empty:
            return
        df = char_df.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.barh(df["character"], df["count"])  # не задаємо кольори
        plt.gca().invert_yaxis()
        plt.xlabel("Mentions")
        plt.title("Top Characters")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    
    def plot_action_graph(self, G: nx.DiGraph, out_path: str):
        """Визуализация графа действий"""
        if G.number_of_nodes() == 0:
            return
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42, k=0.6)
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=1800,
            font_size=10,
            font_weight='bold',
            node_color="#b3d9ff",
            edgecolors="black",
            linewidths=0.8,
        )
        labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="purple")
        plt.title("Action Graph (who → what → whom)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


    # ---------- End-to-end ----------
    def analyze(self, text: str, out_dir: str) -> Dict[str, str]:
        doc = self.read_text(text)
        # Characters
        char_df = self.extract_characters(doc)
        char_csv = f"{out_dir}/characters.csv"
        char_df.to_csv(char_csv, index=False)
        self.plot_top_characters(char_df, f"{out_dir}/characters.png")
        # SVO
        triples = self.extract_svo(doc)
        svo_csv = f"{out_dir}/actions.csv"
        pd.DataFrame([t.__dict__ for t in triples]).to_csv(svo_csv, index=False)
        G = self.build_action_graph(triples)
        self.plot_action_graph(G, f"{out_dir}/actions_graph.png")
        # Emotions
        curve = self.emotion_curve(doc)
        emo_csv = f"{out_dir}/emotion_curve.csv"
        curve.to_csv(emo_csv, index=False)
        self.plot_emotion_curve(curve, f"{out_dir}/emotion_curve.png")
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