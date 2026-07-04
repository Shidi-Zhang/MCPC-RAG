import argparse
import ast
from collections import Counter
import json
import os
import re
import statistics
import time
from typing import Dict, List
from pathlib import Path

import pandas as pd
import requests
from langchain.embeddings.base import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from prompt import (
    LLM_extract_focus_keywords_MIMIC,
    apply_qwen_prefix,
    call_LLM_doctor_summary,
    call_LLM_rag_tendency_analyzer_MIMIC,
    call_LLM_simple_prescription_with_reason_prompt_MIMIC,
    parse_json_garbage,
    parse_model_output,
    parse_prescription_to_list,
    strip_think_tags,
)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain embedding wrapper that loads the local MiniLM model first."""

    def __init__(self, model_path: Path):
        self.model = SentenceTransformer(str(model_path))

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


embedding_model = LocalSentenceTransformerEmbeddings(Path(__file__).resolve().parent / "all-MiniLM-L6-v2")

DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OPENAI_BASE_URL = "https://api.bitidea.cn/v1/chat/completions"
DEFAULT_OPENAI_API_KEY_ENV = "BITIDEA_API_KEY"


class OpenAICompatibleChatLLM(Runnable):
    """Minimal LangChain-compatible caller for OpenAI-style chat/completions APIs."""

    def __init__(
        self,
        model,
        api_key,
        base_url,
        temperature=0.0,
        seed=None,
        timeout=180,
        max_retries=4,
        pause_seconds=0.0,
        max_tokens=None,
    ):
        self.model = model
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause_seconds = pause_seconds
        self.max_tokens = max_tokens

    @staticmethod
    def _message_role(message):
        role = getattr(message, "type", None) or getattr(message, "role", None) or "user"
        return {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "chat": "user",
        }.get(str(role), str(role))

    def _to_messages(self, prompt_input):
        if hasattr(prompt_input, "to_messages"):
            raw_messages = prompt_input.to_messages()
        elif isinstance(prompt_input, list):
            raw_messages = prompt_input
        else:
            raw_messages = [prompt_input]

        messages = []
        for message in raw_messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
            else:
                role = self._message_role(message)
                content = getattr(message, "content", message)
            messages.append({"role": role, "content": str(content)})
        return messages

    @staticmethod
    def _extract_content(response_payload):
        if isinstance(response_payload, str):
            try:
                response_payload = json.loads(response_payload)
            except Exception:
                return response_payload
        if isinstance(response_payload, list):
            parts = []
            for item in response_payload:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        parts.append(str(text))
                elif item:
                    parts.append(str(item))
            return "\n".join(parts) if parts else json.dumps(response_payload, ensure_ascii=False)
        if not isinstance(response_payload, dict):
            return str(response_payload)

        choices = response_payload.get("choices") or []
        if choices:
            first = choices[0] or {}
            message = first.get("message") or {}
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message.get("content"))
            if first.get("text") is not None:
                return str(first.get("text"))

        for key in ("content", "response", "output", "answer"):
            if response_payload.get(key) is not None:
                value = response_payload.get(key)
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return json.dumps(response_payload, ensure_ascii=False)

    def invoke(self, input, config=None, **kwargs):
        if self.pause_seconds and self.pause_seconds > 0:
            time.sleep(float(self.pause_seconds))
        payload = {
            "model": self.model,
            "messages": self._to_messages(input),
            "temperature": self.temperature,
        }
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = int(self.max_tokens)
        if self.seed is not None:
            payload["seed"] = self.seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    data = response.text
                return self._extract_content(data)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible chat/completions call failed: {last_error}")


class MessagesCompatibleChatLLM(OpenAICompatibleChatLLM):
    """Minimal caller for OpenAI/Anthropic-like /v1/messages APIs."""

    @staticmethod
    def _split_system_messages(messages):
        system_parts = []
        conversation = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conversation.append({"role": "assistant", "content": content})
            else:
                conversation.append({"role": "user", "content": content})
        if not conversation:
            conversation = [{"role": "user", "content": "\n".join(system_parts) or ""}]
            system_parts = []
        return "\n".join(part for part in system_parts if part), conversation

    @staticmethod
    def _extract_content(response_payload):
        if isinstance(response_payload, str):
            try:
                response_payload = json.loads(response_payload)
            except Exception:
                return response_payload
        if isinstance(response_payload, list):
            return OpenAICompatibleChatLLM._extract_content(response_payload)
        if not isinstance(response_payload, dict):
            return str(response_payload)

        content = response_payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text is not None:
                        parts.append(str(text))
                elif item is not None:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)
        message = response_payload.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message.get("content"))
        return OpenAICompatibleChatLLM._extract_content(response_payload)

    def invoke(self, input, config=None, **kwargs):
        if self.pause_seconds and self.pause_seconds > 0:
            time.sleep(float(self.pause_seconds))
        system_text, conversation = self._split_system_messages(self._to_messages(input))
        payload = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "max_tokens": int(self.max_tokens) if self.max_tokens and self.max_tokens > 0 else 2048,
        }
        if system_text:
            payload["system"] = system_text
        if self.seed is not None:
            payload["seed"] = self.seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    data = response.text
                return self._extract_content(data)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Messages-compatible /v1/messages call failed: {last_error}")


def load_atc_semantics(path):
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        print(f"[WARN] ATC semantics CSV not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "atc_code" not in df.columns:
        print(f"[WARN] ATC semantics CSV missing atc_code column: {csv_path}")
        return {}
    semantics = {}
    for row in df.to_dict("records"):
        code = str(row.get("atc_code", "")).strip()
        if not code:
            continue
        semantics[code] = {
            "atc_name": str(row.get("atc_name", "")).strip(),
            "top_drug_names": str(row.get("top_drug_names", "")).strip(),
            "total_count": str(row.get("total_count", "")).strip(),
        }
    print(f"[INFO] Loaded ATC semantics: {len(semantics):,} codes from {csv_path}")
    return semantics


def describe_medication_code(code, atc_semantics=None):
    text = str(code).strip()
    if not text:
        return ""
    info = (atc_semantics or {}).get(text, {})
    atc_name = str(info.get("atc_name", "") or "").strip()
    top_drug_names = str(info.get("top_drug_names", "") or "").strip()
    parts = [text]
    details = []
    if atc_name and atc_name != text:
        details.append(atc_name)
    if top_drug_names:
        details.append(f"common MIMIC drugs: {top_drug_names}")
    if details:
        parts.append(f"({'; '.join(details)})")
    return " ".join(parts)


def format_medication_list_for_llm(codes, atc_semantics=None, max_items=30):
    items = []
    for code in (codes or [])[:max_items]:
        desc = describe_medication_code(code, atc_semantics)
        if desc:
            items.append(desc)
    return ", ".join(items) if items else "None"


LLM_CANDIDATE_DELTA_VERIFIER_PROMPT_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a strict Clinical Auditor for MIMIC-IV medication recommendation.

Your task is to refine a draft medication plan using patient context, RAG evidence, and a CANDIDATE MEDICATION SET.

STRICT RULES:
1. Final medications MUST be selected only from Candidate Medication Set.
2. Do NOT invent medication classes outside the candidate set.
3. Keep Active History medications if they remain clinically reasonable.
4. Remove draft medications unsupported by diagnoses, active history, or RAG evidence.
5. Add candidate medications only when current diagnoses or similar cases support them.
6. Use exact medication strings from Candidate Medication Set. Do not rename, expand, translate, or paraphrase ATC codes.
7. Prefer clinically adequate coverage over an overly tiny set. Do not drop well-supported candidate classes only because the set is moderately large.

Return JSON only:
{{
  "final_prescription": ["candidate_med_1", "candidate_med_2"],
  "audit_log": [
    {{"action": "KEPT|ADDED|REMOVED", "drug": "drug", "reason": "short reason"}}
  ],
  "final_description": "short summary"
}}
No markdown, no extra keys.
"""),
    ("user", """Current rich patient context:
{patient_context}

Current diagnoses:
{diagnoses}

Recent visit history:
{recent_visit_history_text}

Active History:
{active_history}

Initial Draft:
{initial_prescription}

RAG focus tendency:
{rag_focus_tendency}

Candidate Medication Set:
{candidate_medications}

Refine the draft under the candidate constraint. Return JSON only.""")
])


LLM_VOTE_GUIDED_CALIBRATOR_PROMPT_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a lightweight clinical medication calibrator for MIMIC-IV medication recommendation.

You are NOT the main decision maker. The main decision comes from RAG vote ranking over similar patients.
Your job is only to make small, conservative adjustments to the RAG-voted candidate list.

STRICT RULES:
1. Final medications MUST come from the RAG Vote Candidate List.
2. Keep all Protected High-Vote Medications unless there is an obvious contradiction.
3. Do not replace high-vote ATC codes with broader categories or invented names.
4. Do not shrink the list aggressively. This is a multi-label medication recommendation task.
5. If uncertain, keep the high-vote medication rather than deleting it.
6. You may remove only candidates that are clearly irrelevant to the diagnoses and recent history.
7. You may add back lower-ranked candidates only if they are in the RAG Vote Candidate List.

Return JSON only:
{{
  "final_prescription": ["ATC_code_1", "ATC_code_2"],
  "calibration_log": [
    {{"action": "KEPT|REMOVED|ADDED", "drug": "ATC_code", "reason": "short reason"}}
  ],
  "final_description": "short summary"
}}
No markdown, no extra keys.
"""),
    ("user", """Current rich patient context:
{patient_context}

Current diagnoses:
{diagnoses}

Recent visit history:
{recent_visit_history_text}

Active History:
{active_history}

Protected High-Vote Medications:
{protected_medications}

RAG Vote Candidate List:
{candidate_medications}

RAG Vote Scores:
{candidate_scores}

Calibrate the RAG-voted list without aggressive deletion. Return JSON only.""")
])


LLM_VOTE_TAIL_PRUNER_PROMPT_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a conservative tail-pruning assistant for medication recommendation.

RAG vote is the main decision maker. You only inspect LOW-PRIORITY tail candidates.

Rules:
1. Protected medications are already selected. Never remove them.
2. You may remove at most {max_remove} tail candidates.
3. Remove only if clearly unrelated to the diagnoses and history.
4. Do not be overly conservative: weak tail candidates reduce Jaccard by adding false positives.
5. If a tail candidate has no direct support from diagnoses, procedures, labs, active history, or high RAG score, remove it.
6. Return JSON only: {{"remove": ["ATC1", "ATC2"], "reason": "short reason"}}
"""),
    ("user", """Current patient context:
{patient_context}

Recent history: {recent_visit_history_text}
Protected selected medications: {protected_medications}
Tail candidates you may remove from: {tail_candidates}

Target final size after pruning: at least {min_final_meds}, with at most {max_remove} removals.

Return JSON only.""")
])


LLM_RAG_RANKED_SELECTOR_PROMPT_MIMIC = ChatPromptTemplate.from_messages([
    ("system", """You are a precision-recall calibrated ATC medication selector for MIMIC-IV medication label prediction.

Goal: maximize Jaccard and F1, not just recall. False positives hurt Jaccard, so do not keep weak candidates merely because they are plausible.

You receive a RAG vote-ranked candidate list. Higher rank and higher vote score are strong statistical priors, but not ground truth.

Decision rules:
1. Select final medications ONLY from the provided RAG candidate list.
2. Use exact ATC codes exactly as written. Do not rename, expand, translate, or invent codes.
3. Treat Protected high-vote candidates as mandatory anchor labels. Keep them unless they are absent from the provided candidate list.
4. For every candidate, require at least one support signal: diagnosis/procedure/lab relevance, active/recent medication history, or strong RAG vote score.
5. Remove candidates that are generic, weakly related, or only remotely plausible. Removing false positives is important for Jaccard.
6. Estimate the target label count primarily from RAG evidence density, not from clinical severity alone:
   - If active history is empty/short and vote scores drop sharply after the first few codes, choose a compact set even for ICU-looking cases.
   - If active history is long, recent visits list many medications, or vote scores remain dense across many ranks, choose a larger set.
   - Do not add broad ICU/supportive medications solely because the patient is critically ill; require RAG or context support for each code.
7. Size guide:
   - sparse evidence / first admission / sharp score drop: usually 2-7 medications
   - moderate evidence density: usually 8-14 medications
   - dense evidence plus long active history or recurrent complex care: usually 15-24 medications
8. Prefer a set that preserves RAG vote recall while trimming only weak tail candidates. Stay within the requested final-size range.
9. If the evidence is ambiguous, use RAG rank as tie-breaker: keep higher-ranked candidates and drop lower-ranked candidates.

Return JSON only:
{{
  "final_prescription": ["ATC_code_1", "ATC_code_2"],
  "estimated_target_size": 8,
  "calibration_log": [
    {{"action": "KEPT|REMOVED", "drug": "ATC_code", "reason": "short evidence-based reason"}}
  ],
  "final_description": "short summary of the precision-recall tradeoff"
}}
No markdown, no extra keys.
"""),
    ("user", """Current patient context:
{patient_context}

Recent visit history:
{recent_visit_history_text}

Active history medications:
{active_history}

Protected high-vote candidates:
{protected_medications}

RAG candidate list with vote scores and ATC semantics:
{ranked_candidates}

RAG score summary:
{score_summary}

Allowed final-size range:
- minimum: {min_final_meds}
- maximum: {max_final_meds}

First estimate the appropriate target size from RAG evidence density and patient context, then select the final ATC set to maximize Jaccard/F1. Return JSON only.""")
])


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_list_cell(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [p.strip() for p in text.split(",") if p.strip()]


def normalize_drug_name_mimic(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return " ".join(cleaned.split())


def list_medicine_answer(answer, ground_truth_list):
    if isinstance(answer, dict):
        value = answer.get("final_prescription") or answer.get("final_prescription_list") or []
        model_drugs_list = [str(x).strip() for x in value] if isinstance(value, list) else []
    elif isinstance(answer, list):
        model_drugs_list = [str(x).strip() for x in answer]
    else:
        model_drugs_list = parse_model_output(answer)

    gt_norm = {normalize_drug_name_mimic(s) for s in ground_truth_list if normalize_drug_name_mimic(s)}
    pred_norm = {normalize_drug_name_mimic(s) for s in model_drugs_list if normalize_drug_name_mimic(s)}

    tp_norm = gt_norm & pred_norm
    fn_norm = gt_norm - pred_norm
    fp_norm = pred_norm - gt_norm

    return {
        "ground_truth_list": ground_truth_list,
        "model_response_answer": model_drugs_list,
        "TruePositive": [gt for gt in ground_truth_list if normalize_drug_name_mimic(gt) in tp_norm],
        "FalseNegative": [gt for gt in ground_truth_list if normalize_drug_name_mimic(gt) in fn_norm],
        "FalsePositive": [pred for pred in model_drugs_list if normalize_drug_name_mimic(pred) in fp_norm],
    }


def compute_scores(metrics):
    tp = len(metrics.get("TruePositive", []))
    fp = len(metrics.get("FalsePositive", []))
    fn = len(metrics.get("FalseNegative", []))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def compute_jaccard(answer, ground_truth_list):
    if isinstance(answer, dict):
        pred = answer.get("final_prescription") or answer.get("final_prescription_list") or []
    elif isinstance(answer, list):
        pred = answer
    else:
        pred = parse_model_output(answer)
    pred_norm = {normalize_drug_name_mimic(s) for s in pred if normalize_drug_name_mimic(str(s))}
    gt_norm = {normalize_drug_name_mimic(s) for s in ground_truth_list if normalize_drug_name_mimic(str(s))}
    union = pred_norm | gt_norm
    if not union:
        return 0.0
    return len(pred_norm & gt_norm) / len(union)


def set_overlap_stats(pred_list, gt_list):
    pred_norm = {normalize_drug_name_mimic(s) for s in pred_list if normalize_drug_name_mimic(str(s))}
    gt_norm = {normalize_drug_name_mimic(s) for s in gt_list if normalize_drug_name_mimic(str(s))}
    tp = pred_norm & gt_norm
    return {
        "pred_size": len(pred_norm),
        "gt_size": len(gt_norm),
        "hit_size": len(tp),
        "recall": len(tp) / len(gt_norm) if gt_norm else 0.0,
        "precision": len(tp) / len(pred_norm) if pred_norm else 0.0,
    }


def _dedup_keep_order(items):
    seen = set()
    out = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _format_recent_visit_history_text(recent_visit_history, max_visits=3):
    if not recent_visit_history:
        return "None"
    lines = []
    for visit in recent_visit_history[-max_visits:]:
        if not isinstance(visit, dict):
            continue
        lines.append(
            f"- {visit.get('visit', 'Visit')}: symptoms={visit.get('symptoms', 'None')}; "
            f"prescription={visit.get('prescription', [])}"
        )
    return "\n".join(lines) if lines else "None"


def _format_focus_tendency(focus_items, max_items=6):
    if not focus_items or not isinstance(focus_items, list):
        return "None"
    lines = []
    for item in focus_items[:max_items]:
        if not isinstance(item, dict):
            continue
        focus = item.get("focus")
        tendency = item.get("tendency", {})
        if not focus:
            continue
        pattern = tendency.get("dominant_pattern", "N/A") if isinstance(tendency, dict) else "N/A"
        additions = tendency.get("common_additions", []) if isinstance(tendency, dict) else []
        reasoning = tendency.get("reasoning") if isinstance(tendency, dict) else ""
        if not isinstance(additions, list):
            additions = [str(additions)] if additions else []
        add_txt = ", ".join(str(x).strip() for x in additions[:4] if str(x).strip()) or "None"
        lines.append(f"- Focus: {focus} | Pattern: {pattern} | Add: {add_txt} | Reasoning: {reasoning or 'N/A'}")
    return "\n".join(lines) if lines else "None"


def build_candidate_medications(active_history, rag_patients, rag_tendency_by_focus, initial_prescription=None, include_initial_draft=False):
    candidates = []
    candidates.extend(active_history or [])
    for patient in rag_patients or []:
        meds = patient.get("medications", []) if isinstance(patient, dict) else []
        if isinstance(meds, str):
            meds = parse_list_cell(meds)
        candidates.extend(meds if isinstance(meds, list) else [])
    for item in rag_tendency_by_focus or []:
        tendency = item.get("tendency", {}) if isinstance(item, dict) else {}
        additions = tendency.get("common_additions", []) if isinstance(tendency, dict) else []
        if isinstance(additions, list):
            candidates.extend(additions)
    if include_initial_draft and initial_prescription:
        candidates.extend(parse_prescription_to_list(initial_prescription))
    return _dedup_keep_order(candidates)


def rank_candidate_medications(active_history, rag_patients, rag_tendency_by_focus, initial_prescription=None, include_initial_draft=False):
    scores = {}
    order = []

    def add(med, score):
        text = str(med).strip()
        if not text:
            return
        if text not in scores:
            scores[text] = 0.0
            order.append(text)
        scores[text] += float(score)

    for med in active_history or []:
        add(med, 5.0)

    for idx, patient in enumerate(rag_patients or []):
        if not isinstance(patient, dict):
            continue
        relevance = safe_float(patient.get("score", 0.0), default=0.0)
        rank_weight = 1.0 / (idx + 1)
        weight = 1.0 + relevance + rank_weight
        meds = patient.get("medications", [])
        if isinstance(meds, str):
            meds = parse_list_cell(meds)
        for med in meds if isinstance(meds, list) else []:
            add(med, weight)

    for item in rag_tendency_by_focus or []:
        tendency = item.get("tendency", {}) if isinstance(item, dict) else {}
        additions = tendency.get("common_additions", []) if isinstance(tendency, dict) else []
        if isinstance(additions, list):
            for med in additions:
                add(med, 2.0)

    if include_initial_draft and initial_prescription:
        for med in parse_prescription_to_list(initial_prescription):
            add(med, 1.5)

    ranked = sorted(order, key=lambda med: (-scores.get(med, 0.0), order.index(med)))
    return ranked, scores


def build_clinical_state_query(row, diagnoses, include_medication_history=True):
    clinical_state = str(row.get("clinical_state", "") or "").strip()
    if clinical_state and clinical_state.lower() != "nan":
        if not include_medication_history:
            lines = [
                line for line in clinical_state.splitlines()
                if not line.strip().lower().startswith("medication history labels:")
            ]
            return "\n".join(lines).strip()
        return clinical_state
    parts = [f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}"]
    procedures = parse_list_cell(row.get("procedures")) if "procedures" in row else []
    lab_summary = parse_list_cell(row.get("lab_summary")) if "lab_summary" in row else []
    if procedures:
        parts.append(f"Procedures: {', '.join(procedures)}")
    if lab_summary:
        parts.append(f"Lab summary: {'; '.join(lab_summary)}")
    return "\n".join(parts)


def _format_context_list_field(row, field_name, label, max_items=18):
    if field_name not in row:
        return ""
    values = parse_list_cell(row.get(field_name))
    if not values:
        return ""
    shown = values[:max_items]
    suffix = f"; ... (+{len(values) - max_items} more)" if len(values) > max_items else ""
    return f"{label}: {'; '.join(shown)}{suffix}"


def _strip_medication_history_from_context(text):
    lines = []
    for line in str(text or "").splitlines():
        if line.strip().lower().startswith("medication history labels:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def build_patient_context_for_llm(row, diagnoses, include_medication_history=False, max_chars=6000):
    """Build a dataset-tolerant context shared by MIMIC-IV, eICU, and MIMIC-III.

    Common fields are always preferred: diagnoses, procedures, and lab_summary.
    Optional fields such as vital_summary and nurse_charting_summary are included
    only when the current dataset has them.
    """
    parts = []
    admittime = str(row.get("admittime", "") or "").strip()
    if admittime:
        parts.append(f"Admission time: {admittime}")
    source = str(row.get("source", "") or "").strip()
    if source and source.lower() != "nan":
        parts.append(f"Dataset source: {source}")

    parts.append(f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}")
    for field_name, label, limit in (
        ("procedures", "Procedures", 18),
        ("lab_summary", "Lab summary", 24),
        ("vital_summary", "Vital summary", 18),
        ("nurse_charting_summary", "Nurse charting summary", 18),
    ):
        text = _format_context_list_field(row, field_name, label, max_items=limit)
        if text:
            parts.append(text)

    clinical_state = str(row.get("clinical_state", "") or "").strip()
    if clinical_state and clinical_state.lower() != "nan":
        if not include_medication_history:
            clinical_state = _strip_medication_history_from_context(clinical_state)
        if clinical_state:
            parts.append(f"Original clinical_state:\n{clinical_state}")

    context = "\n".join(part for part in parts if part).strip()
    return context[:max_chars] if context else f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}"


def build_retrieved_case_context(metadata, max_chars=1800):
    clinical_state = _strip_medication_history_from_context(metadata.get("clinical_state", ""))
    if clinical_state:
        return clinical_state[:max_chars]
    parts = [
        f"Diagnoses: {', '.join(metadata.get('diagnoses', []) or []) or 'None'}",
    ]
    for field_name, label, limit in (
        ("procedures", "Procedures", 10),
        ("lab_summary", "Lab summary", 12),
        ("vital_summary", "Vital summary", 8),
        ("nurse_charting_summary", "Nurse charting summary", 8),
    ):
        values = metadata.get(field_name, []) or []
        if isinstance(values, str):
            values = parse_list_cell(values)
        if values:
            parts.append(f"{label}: {'; '.join([str(x) for x in values[:limit]])}")
    return "\n".join(parts)[:max_chars]


def build_rag_hit_payload(doc, score):
    metadata = doc.metadata or {}
    embedded_demographics = parse_demographics_from_clinical_state(metadata.get("clinical_state", ""))
    payload = {
        "source": metadata.get("source", ""),
        "subject_id": metadata.get("subject_id"),
        "hadm_id": metadata.get("hadm_id"),
        "hospitalid": metadata.get("hospitalid", embedded_demographics.get("hospitalid", "")),
        "score": safe_float(score),
        "diagnoses": metadata.get("diagnoses", []),
        "medications": metadata.get("medications", []),
        "procedures": metadata.get("procedures", []),
        "lab_summary": metadata.get("lab_summary", []),
        "vital_summary": metadata.get("vital_summary", []),
        "nurse_charting_summary": metadata.get("nurse_charting_summary", []),
        "clinical_state": metadata.get("clinical_state", ""),
        "content": doc.page_content,
    }
    for key, value in embedded_demographics.items():
        payload.setdefault(key, value)
    for key in (
        "age",
        "anchor_age",
        "admission_age",
        "gender",
        "sex",
        "ethnicity",
        "race",
        "weight",
        "height",
        "unit",
        "admission_type",
        "admissiontype",
        "icu_type",
        "unittype",
        "careunit",
        "first_careunit",
        "admission_location",
        "admissionsource",
    ):
        if key in metadata:
            payload[key] = metadata.get(key)
    return payload


def count_sources(items):
    counts = Counter(str((item or {}).get("source", "") or "unknown") for item in items or [])
    return dict(counts)


FIELD_AWARE_DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "diagnoses": 0.30,
    "procedures": 0.08,
    "labs": 0.16,
    "vitals": 0.08,
    "demographics": 0.03,
    "nursing": 0.02,
    "context": 0.03,
}


def parse_field_weights(text):
    weights = dict(FIELD_AWARE_DEFAULT_WEIGHTS)
    if not text:
        return weights
    for item in str(text).split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in weights:
            continue
        try:
            weights[key] = max(0.0, float(value))
        except Exception:
            continue
    return weights


def _normalize_token(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _field_tokens(values):
    if isinstance(values, str):
        values = parse_list_cell(values)
    tokens = set()
    for value in values or []:
        norm = _normalize_token(value)
        for token in norm.split():
            if len(token) >= 2 and token not in {"none", "nan", "and", "the", "with"}:
                tokens.add(token)
    return tokens


def jaccard_similarity(left, right):
    left_tokens = _field_tokens(left)
    right_tokens = _field_tokens(right)
    if not left_tokens or not right_tokens:
        return None
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def extract_named_numeric_values(values):
    if isinstance(values, str):
        values = parse_list_cell(values)
    extracted = {}
    for item in values or []:
        text = str(item)
        name = text.split(":", 1)[0].strip().lower() if ":" in text else text.strip().lower()
        name = re.sub(r"^(abnormal|high|low)\s+", "", name)
        nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
        if not name or not nums:
            continue
        if "mean" in text.lower() and len(nums) >= 3:
            value = nums[-1]
        else:
            value = sum(nums) / len(nums)
        extracted[name] = value
    return extracted


def numeric_field_similarity(left, right, scale=10.0):
    left_values = extract_named_numeric_values(left)
    right_values = extract_named_numeric_values(right)
    common = sorted(set(left_values) & set(right_values))
    if not common:
        return None
    scores = []
    for name in common:
        diff = abs(left_values[name] - right_values[name])
        denom = max(scale, abs(left_values[name]), abs(right_values[name]), 1.0)
        scores.append(max(0.0, 1.0 - diff / denom))
    return sum(scores) / len(scores) if scores else None


def first_present_value(mapping, names):
    for name in names:
        if name in mapping:
            value = mapping.get(name)
            if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
                return value
    return None


def parse_demographics_from_clinical_state(text):
    text = str(text or "")
    match = re.search(r"^Demographics:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return {}
    values = {}
    for item in match.group(1).split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            continue
        key_map = {
            "hospital": "hospitalid",
            "unit": "unit",
        }
        values[key_map.get(key, key)] = value
    return values


def enrich_with_embedded_demographics(mapping):
    values = {}
    if isinstance(mapping, dict):
        values.update(mapping)
    else:
        try:
            values.update(mapping.to_dict())
        except Exception:
            pass
    embedded = parse_demographics_from_clinical_state(values.get("clinical_state", ""))
    for key, value in embedded.items():
        values.setdefault(key, value)
    return values


def safe_age(value):
    try:
        age = float(value)
        if 0 <= age <= 120:
            return age
    except Exception:
        pass
    return None


def safe_optional_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return float(text)
    except Exception:
        return None


def demographic_similarity(query_row, candidate_metadata):
    query_row = enrich_with_embedded_demographics(query_row)
    candidate_metadata = enrich_with_embedded_demographics(candidate_metadata)
    signals = []
    q_age = safe_age(first_present_value(query_row, ["age", "anchor_age", "admission_age"]))
    c_age = safe_age(first_present_value(candidate_metadata, ["age", "anchor_age", "admission_age"]))
    if q_age is not None and c_age is not None:
        signals.append(pow(2.718281828, -abs(q_age - c_age) / 20.0))

    for names, scale in ((["weight"], 30.0), (["height"], 20.0)):
        q_val = safe_optional_float(first_present_value(query_row, names))
        c_val = safe_optional_float(first_present_value(candidate_metadata, names))
        if q_val is not None and c_val is not None and q_val > 0 and c_val > 0:
            signals.append(pow(2.718281828, -abs(q_val - c_val) / scale))

    for names in (["gender", "sex"], ["ethnicity", "race"]):
        q_val = first_present_value(query_row, names)
        c_val = first_present_value(candidate_metadata, names)
        if q_val is not None and c_val is not None:
            signals.append(1.0 if _normalize_token(q_val) == _normalize_token(c_val) else 0.0)
    if not signals:
        return None
    return sum(signals) / len(signals)


def context_similarity(query_row, candidate_metadata):
    query_row = enrich_with_embedded_demographics(query_row)
    candidate_metadata = enrich_with_embedded_demographics(candidate_metadata)
    signals = []
    for names in (
        ["admission_type", "admissiontype"],
        ["icu_type", "unittype", "careunit", "first_careunit", "unit"],
        ["admission_location", "admissionsource"],
    ):
        q_val = first_present_value(query_row, names)
        c_val = first_present_value(candidate_metadata, names)
        if q_val is not None and c_val is not None:
            signals.append(1.0 if _normalize_token(q_val) == _normalize_token(c_val) else 0.0)
    if not signals:
        return None
    return sum(signals) / len(signals)


def field_aware_score(query_row, candidate_payload, semantic_score, weights):
    candidate_metadata = candidate_payload or {}
    query_diagnoses = parse_list_cell(query_row.get("diagnoses"))
    scores = {
        "semantic": safe_float(semantic_score, default=0.0),
        "diagnoses": jaccard_similarity(query_diagnoses, candidate_metadata.get("diagnoses", [])),
        "procedures": jaccard_similarity(parse_list_cell(query_row.get("procedures")), candidate_metadata.get("procedures", [])),
        "labs": numeric_field_similarity(parse_list_cell(query_row.get("lab_summary")), candidate_metadata.get("lab_summary", [])),
        "vitals": numeric_field_similarity(parse_list_cell(query_row.get("vital_summary")), candidate_metadata.get("vital_summary", []), scale=20.0),
        "nursing": jaccard_similarity(parse_list_cell(query_row.get("nurse_charting_summary")), candidate_metadata.get("nurse_charting_summary", [])),
        "demographics": demographic_similarity(query_row, candidate_metadata),
        "context": context_similarity(query_row, candidate_metadata),
    }
    numerator = 0.0
    denominator = 0.0
    effective_fields = {}
    for field, score in scores.items():
        if score is None:
            continue
        weight = float(weights.get(field, 0.0))
        if weight <= 0:
            continue
        clipped = max(0.0, min(1.0, float(score)))
        numerator += weight * clipped
        denominator += weight
        effective_fields[field] = round(clipped, 4)
    if denominator <= 0:
        return safe_float(semantic_score, default=0.0), effective_fields
    return numerator / denominator, effective_fields


def apply_final_strategy(final_result, ranked_candidates, strategy, vote_top_n=0, min_final_meds=0, max_final_meds=0):
    if strategy == "verifier":
        return final_result

    verifier_meds = []
    if isinstance(final_result, dict):
        verifier_meds = final_result.get("final_prescription") or []
    verifier_meds = [str(med).strip() for med in verifier_meds if str(med).strip()]
    vote_n = vote_top_n if vote_top_n and vote_top_n > 0 else 10
    vote_meds = ranked_candidates[:vote_n]

    if strategy == "rag_vote":
        merged = vote_meds
    elif strategy == "hybrid":
        target_min = min_final_meds if min_final_meds and min_final_meds > 0 else min(8, len(ranked_candidates))
        merged = _dedup_keep_order(verifier_meds)
        for med in vote_meds:
            if len(merged) >= target_min:
                break
            merged.append(med)
        merged = _dedup_keep_order(merged)
    else:
        merged = verifier_meds

    if max_final_meds and max_final_meds > 0:
        merged = merged[:max_final_meds]

    if not isinstance(final_result, dict):
        final_result = {}
    updated = dict(final_result)
    updated["final_prescription"] = _dedup_keep_order(merged)
    updated["selection_strategy"] = strategy
    updated["rag_vote_top"] = vote_meds
    return updated


def _rank_ordered_subset(items, ranked_candidates, limit=None):
    item_set = set(str(item).strip() for item in items if str(item).strip())
    ordered = [med for med in ranked_candidates if med in item_set]
    seen = set(ordered)
    for med in items:
        text = str(med).strip()
        if text and text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered[:limit] if limit and limit > 0 else ordered


def _size_guard_features(ranked_candidates, candidate_vote_scores, active_history, rag_patients, vote_top_n=30):
    candidate_pool = ranked_candidates[:vote_top_n] if vote_top_n and vote_top_n > 0 else ranked_candidates[:30]
    scores = [
        float(candidate_vote_scores.get(med, 0.0) if isinstance(candidate_vote_scores, dict) else 0.0)
        for med in candidate_pool
    ]
    top_score = scores[0] if scores else 0.0

    def count_ratio(ratio):
        if top_score <= 0:
            return 0
        return sum(1 for score in scores if score >= ratio * top_score)

    rag_sizes = []
    for patient in (rag_patients or [])[:10]:
        meds = patient.get("medications", []) if isinstance(patient, dict) else []
        if isinstance(meds, str):
            meds = parse_list_cell(meds)
        if isinstance(meds, list) and meds:
            rag_sizes.append(len(_dedup_keep_order(meds)))

    return {
        "active_history_size": len(_dedup_keep_order(active_history or [])),
        "rag_top5_median_size": statistics.median(rag_sizes[:5]) if rag_sizes else 0,
        "rag_top10_median_size": statistics.median(rag_sizes) if rag_sizes else 0,
        "count_score_gte_70pct_top": count_ratio(0.70),
        "count_score_gte_40pct_top": count_ratio(0.40),
        "count_score_gte_20pct_top": count_ratio(0.20),
        "candidate_pool_size": len(candidate_pool),
    }


def apply_size_guard_postprocess(
    final_result,
    ranked_candidates,
    candidate_vote_scores,
    active_history,
    rag_patients,
    vote_top_n=30,
    max_final_meds=22,
):
    if not isinstance(final_result, dict):
        return final_result

    final = _dedup_keep_order(final_result.get("final_prescription") or [])
    if not final:
        return final_result

    candidate_pool = ranked_candidates[:vote_top_n] if vote_top_n and vote_top_n > 0 else ranked_candidates[:30]
    max_allowed = max_final_meds if max_final_meds and max_final_meds > 0 else min(len(candidate_pool), 22)
    max_allowed = max(1, min(max_allowed, len(candidate_pool) if candidate_pool else max_allowed))
    features = _size_guard_features(
        ranked_candidates=ranked_candidates,
        candidate_vote_scores=candidate_vote_scores,
        active_history=active_history,
        rag_patients=rag_patients,
        vote_top_n=vote_top_n,
    )

    active_count = features["active_history_size"]
    rag_top5 = features["rag_top5_median_size"]
    count_70 = features["count_score_gte_70pct_top"]
    count_40 = features["count_score_gte_40pct_top"]

    small_case = active_count == 0 and count_70 <= 5 and (rag_top5 <= 8 or count_40 <= 9)
    large_case = (
        not small_case
        and (
            active_count >= 16
            or (rag_top5 >= 20 and count_40 >= 14)
            or count_40 >= 22
        )
    )

    adjusted = list(final)
    action = "none"
    target_size = len(adjusted)
    reason = "No size guard adjustment."

    if small_case and len(adjusted) > 6:
        target_size = 6 if rag_top5 <= 6 and count_70 <= 4 else 8
        target_size = min(target_size, len(adjusted))
        adjusted = _rank_ordered_subset(adjusted, candidate_pool, limit=target_size)
        action = "small_case_trim"
        reason = (
            f"Small-prescription guard trimmed final size to {target_size}: "
            f"active_history=0, rag_top5_median_size={rag_top5}, "
            f"count_score>=70%top={count_70}, count_score>=40%top={count_40}."
        )
    elif large_case and len(adjusted) < max_allowed:
        if active_count >= 16 or count_40 >= 22:
            target_size = max_allowed
        else:
            target_size = min(max_allowed, max(len(adjusted), 20))
        adjusted_set = set(adjusted)
        for med in candidate_pool:
            if len(adjusted) >= target_size:
                break
            if med not in adjusted_set:
                adjusted.append(med)
                adjusted_set.add(med)
        action = "large_case_fill"
        reason = (
            f"Large-prescription guard filled final size to {len(adjusted)}: "
            f"active_history_size={active_count}, rag_top5_median_size={rag_top5}, "
            f"count_score>=40%top={count_40}."
        )

    updated = dict(final_result)
    updated["final_prescription"] = _dedup_keep_order(adjusted)
    updated["size_guard"] = {
        "enabled": True,
        "action": action,
        "target_size": target_size,
        "before_size": len(final),
        "after_size": len(updated["final_prescription"]),
        "reason": reason,
        "features": features,
    }
    if action != "none":
        audit_log = list(updated.get("audit_log") or [])
        audit_log.append({"action": action, "drug": "size_guard", "reason": reason})
        updated["audit_log"] = audit_log
        calibration_log = list(updated.get("calibration_log") or [])
        calibration_log.append({"action": action, "drug": "size_guard", "reason": reason})
        updated["calibration_log"] = calibration_log
    return updated


def _final_payload(final_answer_list, final_answer):
    if isinstance(final_answer, dict):
        value = final_answer.get("final_prescription") or final_answer.get("final_prescription_list")
        if isinstance(value, list):
            return value
    if isinstance(final_answer_list, dict):
        value = final_answer_list.get("model_response_answer")
        if isinstance(value, list):
            return value
    return final_answer_list


def normalize_openai_base_url(url):
    text = str(url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/messages"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def openai_chat_completions_url(url):
    base = normalize_openai_base_url(url)
    return f"{base}/chat/completions" if base else ""


def openai_messages_url(url):
    base = normalize_openai_base_url(url)
    return f"{base}/messages" if base else ""


def make_llm(
    model_name,
    temperature,
    seed,
    num_predict=512,
    json_mode=False,
    provider="auto",
    openai_base_url="",
    openai_api_key_env="OPENAI_API_KEY",
    openai_api_key="",
    llm_call_pause_seconds=0.0,
    ollama_num_ctx=8192,
    ollama_keep_alive="10m",
    ollama_num_thread=0,
    ollama_reasoning=False,
):
    normalized_provider = (provider or "auto").lower()
    normalized_base_url = normalize_openai_base_url(openai_base_url)
    lowered_model = str(model_name).lower()
    openai_like_model = lowered_model.startswith(("gpt", "o1", "o3", "o4"))
    use_messages = normalized_provider == "messages"
    use_openai = normalized_provider == "openai" or (
        normalized_provider == "auto" and (openai_like_model or normalized_base_url)
    )

    if use_openai or use_messages:
        api_key = str(openai_api_key or "").strip() or os.environ.get(openai_api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"Missing OpenAI-compatible API key. Set {openai_api_key_env or 'OPENAI_API_KEY'} "
                "or pass --openai_api_key."
            )
        endpoint = openai_messages_url(openai_base_url) if use_messages else openai_chat_completions_url(openai_base_url)
        if not endpoint:
            raise ValueError("Missing OpenAI-compatible base URL.")
        llm_cls = MessagesCompatibleChatLLM if use_messages else OpenAICompatibleChatLLM
        return llm_cls(
            model=model_name,
            api_key=api_key,
            base_url=endpoint,
            temperature=temperature,
            seed=seed,
            pause_seconds=llm_call_pause_seconds,
            max_tokens=num_predict,
        )

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "seed": seed,
        "num_predict": num_predict,
        "reasoning": ollama_reasoning,
    }
    if ollama_num_ctx and ollama_num_ctx > 0:
        kwargs["num_ctx"] = ollama_num_ctx
    if ollama_keep_alive:
        kwargs["keep_alive"] = ollama_keep_alive
    if ollama_num_thread and ollama_num_thread > 0:
        kwargs["num_thread"] = ollama_num_thread
    if json_mode:
        kwargs["format"] = "json"
    return OllamaLLM(**kwargs)


def strip_json_fences(text):
    clean = str(text or "").strip()
    if "```json" in clean:
        return clean.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in clean:
        return clean.split("```", 1)[1].split("```", 1)[0].strip()
    return clean


def coerce_llm_json_object(parsed):
    if isinstance(parsed, dict):
        expected_keys = {
            "final_prescription",
            "final_prescription_list",
            "calibration_log",
            "audit_log",
            "estimated_target_size",
            "final_description",
            "remove",
            "reason",
        }
        if expected_keys.intersection(parsed.keys()):
            return parsed
        for key in ("text", "content", "output", "response", "answer"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return parse_llm_json_object(value)
            if isinstance(value, (dict, list)):
                return coerce_llm_json_object(value)
        return parsed

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and (
                "final_prescription" in item
                or "final_prescription_list" in item
                or "remove" in item
            ):
                return item

        text_parts = []
        for item in parsed:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value)
            elif isinstance(item, str) and item.strip():
                text_parts.append(item)
        for text in text_parts:
            try:
                return parse_llm_json_object(text)
            except Exception:
                continue

        if parsed and all(not isinstance(item, (dict, list)) for item in parsed):
            return {"final_prescription": [str(item).strip() for item in parsed if str(item).strip()]}

    raise ValueError(f"Expected JSON object with prescription fields, got {type(parsed).__name__}")


def parse_partial_prescription_json(text):
    clean = strip_json_fences(text)
    match = re.search(r'"final_prescription"\s*:\s*(\[[^\]]*\])', clean, flags=re.DOTALL)
    if not match:
        match = re.search(r'"final_prescription_list"\s*:\s*(\[[^\]]*\])', clean, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not find final_prescription array in partial JSON output")

    meds = ast.literal_eval(match.group(1))
    if not isinstance(meds, list):
        raise ValueError("Partial final_prescription is not a list")

    parsed = {"final_prescription": meds}
    size_match = re.search(r'"estimated_target_size"\s*:\s*(\d+)', clean)
    if size_match:
        parsed["estimated_target_size"] = int(size_match.group(1))
    parsed["final_description"] = "Recovered final_prescription from partial/truncated LLM JSON output."
    return parsed


def parse_llm_json_object(text):
    clean = strip_json_fences(text)
    try:
        parsed = parse_json_garbage(clean)
    except Exception:
        try:
            parsed = json.loads(clean)
        except Exception:
            return parse_partial_prescription_json(clean)
    if not isinstance(parsed, (dict, list)):
        try:
            parsed = json.loads(clean)
        except Exception:
            return parse_partial_prescription_json(clean)
    return coerce_llm_json_object(parsed)


def call_candidate_delta_verifier(
    initial_prescription,
    diagnoses,
    active_history,
    llm,
    candidate_medications,
    rag_tendency_by_focus=None,
    recent_visit_history=None,
    patient_context=None,
):
    prompt = apply_qwen_prefix(LLM_CANDIDATE_DELTA_VERIFIER_PROMPT_MIMIC, llm)
    chain = prompt | llm | StrOutputParser()
    candidate_set = set(candidate_medications or [])
    active_history_text = active_history if active_history and len(active_history) > 0 else "None"
    result = chain.invoke({
        "patient_context": patient_context or f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}",
        "diagnoses": ", ".join(diagnoses),
        "active_history": active_history_text,
        "initial_prescription": initial_prescription,
        "rag_focus_tendency": _format_focus_tendency(rag_tendency_by_focus),
        "recent_visit_history_text": _format_recent_visit_history_text(recent_visit_history, max_visits=3),
        "candidate_medications": ", ".join(candidate_medications) if candidate_medications else "None",
    })
    result = strip_think_tags(result).strip()

    try:
        parsed = parse_llm_json_object(result)

        final_prescription = parsed.get("final_prescription", [])
        if isinstance(final_prescription, str):
            final_prescription = [s.strip() for s in final_prescription.split(",")]
        final_prescription = [str(s).split("|")[0].strip() for s in final_prescription if str(s).strip()]
        constrained = [med for med in final_prescription if med in candidate_set]
        if not constrained:
            constrained = [med for med in parse_prescription_to_list(initial_prescription) if med in candidate_set]
        if not constrained and candidate_medications:
            constrained = list(candidate_medications[: min(3, len(candidate_medications))])

        removed_by_constraint = [med for med in final_prescription if med not in candidate_set]
        audit_log = parsed.get("audit_log", [])
        if not isinstance(audit_log, list):
            audit_log = []
        for med in removed_by_constraint:
            audit_log.append({
                "action": "REMOVED",
                "drug": med,
                "reason": "Removed by candidate medication constraint.",
            })

        return {
            "final_prescription": _dedup_keep_order(constrained),
            "audit_log": audit_log,
            "final_description": parsed.get("final_description", ""),
            "raw_output": result,
            "candidate_constraint": {
                "num_candidates": len(candidate_set),
                "removed_out_of_candidate": removed_by_constraint,
            },
        }
    except Exception as exc:
        fallback = [med for med in parse_prescription_to_list(initial_prescription) if med in candidate_set]
        if not fallback and candidate_medications:
            fallback = list(candidate_medications[: min(3, len(candidate_medications))])
        return {
            "final_prescription": _dedup_keep_order(fallback),
            "audit_log": [{
                "action": "KEPT",
                "drug": "candidate_fallback",
                "reason": f"Verifier parse failed; candidate-constrained fallback used: {exc}",
            }],
            "final_description": "Candidate-constrained fallback was used because verifier output could not be parsed.",
            "raw_output": result,
            "candidate_constraint": {
                "num_candidates": len(candidate_set),
                "removed_out_of_candidate": [],
            },
        }


def _format_candidate_scores(candidate_medications, candidate_vote_scores, max_items=30):
    lines = []
    for med in candidate_medications[:max_items]:
        score = candidate_vote_scores.get(med, 0.0) if isinstance(candidate_vote_scores, dict) else 0.0
        lines.append(f"- {med}: {float(score):.4f}")
    return "\n".join(lines) if lines else "None"


def _format_ranked_candidates_for_llm(candidate_medications, candidate_vote_scores, atc_semantics=None, max_items=30):
    lines = []
    for rank, med in enumerate(candidate_medications[:max_items], start=1):
        score = candidate_vote_scores.get(med, 0.0) if isinstance(candidate_vote_scores, dict) else 0.0
        desc = describe_medication_code(med, atc_semantics)
        lines.append(f"{rank}. {desc} | vote_score={float(score):.4f}")
    return "\n".join(lines) if lines else "None"


def _format_score_summary_for_llm(candidate_medications, candidate_vote_scores, max_items=30):
    scored = [
        (med, float(candidate_vote_scores.get(med, 0.0) if isinstance(candidate_vote_scores, dict) else 0.0))
        for med in candidate_medications[:max_items]
    ]
    scores = [score for _, score in scored]
    if not scores:
        return "No candidate vote scores."

    top = scores[0]
    strong_70 = sum(1 for score in scores if top > 0 and score >= 0.70 * top)
    moderate_40 = sum(1 for score in scores if top > 0 and score >= 0.40 * top)
    weak_20 = sum(1 for score in scores if top > 0 and score >= 0.20 * top)
    drops = []
    for idx in range(len(scores) - 1):
        if scores[idx] > 0:
            drops.append((idx + 1, (scores[idx] - scores[idx + 1]) / scores[idx]))
    max_drop_rank, max_drop = max(drops, key=lambda item: item[1]) if drops else (0, 0.0)

    def avg(first_n):
        values = scores[: min(first_n, len(scores))]
        return sum(values) / len(values) if values else 0.0

    return (
        f"top_score={top:.4f}; avg_top5={avg(5):.4f}; avg_top10={avg(10):.4f}; "
        f"avg_top20={avg(20):.4f}; count_score>=70%top={strong_70}; "
        f"count_score>=40%top={moderate_40}; count_score>=20%top={weak_20}; "
        f"largest_relative_drop_after_rank={max_drop_rank} ({max_drop:.2%}). "
        "Use a compact target if strong/moderate counts are small or there is an early large drop."
    )


def call_vote_guided_calibrator(
    diagnoses,
    active_history,
    recent_visit_history,
    llm,
    ranked_candidates,
    candidate_vote_scores,
    patient_context=None,
    protected_top_n=5,
    vote_top_n=12,
    min_final_meds=8,
    max_final_meds=12,
    atc_semantics=None,
):
    candidate_pool = ranked_candidates[:vote_top_n] if vote_top_n and vote_top_n > 0 else ranked_candidates[:12]
    protected = ranked_candidates[:protected_top_n] if protected_top_n and protected_top_n > 0 else []
    candidate_set = set(candidate_pool)
    protected_set = set(protected)
    prompt = apply_qwen_prefix(LLM_VOTE_GUIDED_CALIBRATOR_PROMPT_MIMIC, llm)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "patient_context": patient_context or f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}",
        "diagnoses": ", ".join(diagnoses),
        "active_history": active_history if active_history else "None",
        "recent_visit_history_text": _format_recent_visit_history_text(recent_visit_history, max_visits=3),
        "protected_medications": format_medication_list_for_llm(protected, atc_semantics),
        "candidate_medications": format_medication_list_for_llm(candidate_pool, atc_semantics),
        "candidate_scores": _format_candidate_scores(candidate_pool, candidate_vote_scores),
    })
    result = strip_think_tags(result).strip()

    try:
        parsed = parse_llm_json_object(result)
        llm_meds = parsed.get("final_prescription", [])
        if isinstance(llm_meds, str):
            llm_meds = [s.strip() for s in llm_meds.split(",")]
        llm_meds = [str(med).strip() for med in llm_meds if str(med).strip()]
        constrained = [med for med in llm_meds if med in candidate_set]
        removed_out_of_candidate = [med for med in llm_meds if med not in candidate_set]
        calibration_log = parsed.get("calibration_log", [])
        if not isinstance(calibration_log, list):
            calibration_log = []
    except Exception as exc:
        fallback_n = min_final_meds if min_final_meds and min_final_meds > 0 else min(8, len(candidate_pool))
        constrained = candidate_pool[:fallback_n]
        removed_out_of_candidate = []
        calibration_log = [{
            "action": "KEPT",
            "drug": "rag_vote_fallback",
            "reason": f"Calibrator parse failed; RAG vote fallback used: {exc}",
        }]
        parsed = {"final_description": "RAG vote fallback was used because calibrator output could not be parsed."}

    merged = _dedup_keep_order(protected + constrained)
    target_min = min_final_meds if min_final_meds and min_final_meds > 0 else min(8, len(candidate_pool))
    for med in candidate_pool:
        if len(_dedup_keep_order(merged)) >= target_min:
            break
        if med in set(merged):
            continue
        merged.append(med)
    merged = _dedup_keep_order(merged)
    if max_final_meds and max_final_meds > 0:
        kept_protected = [med for med in protected if med in merged]
        rest = [med for med in merged if med not in set(kept_protected)]
        merged = _dedup_keep_order(kept_protected + rest)[:max_final_meds]

    return {
        "final_prescription": merged,
        "audit_log": calibration_log,
        "calibration_log": calibration_log,
        "final_description": parsed.get("final_description", ""),
        "raw_output": result,
        "selection_strategy": "vote_llm",
        "protected_high_vote_medications": protected,
        "rag_vote_top": candidate_pool,
        "candidate_constraint": {
            "num_candidates": len(candidate_set),
            "removed_out_of_candidate": removed_out_of_candidate,
        },
    }


def call_rag_ranked_selector(
    diagnoses,
    active_history,
    recent_visit_history,
    llm,
    ranked_candidates,
    candidate_vote_scores,
    patient_context,
    protected_top_n=6,
    vote_top_n=20,
    min_final_meds=8,
    max_final_meds=12,
    atc_semantics=None,
):
    candidate_pool = ranked_candidates[:vote_top_n] if vote_top_n and vote_top_n > 0 else ranked_candidates[:20]
    protected = candidate_pool[:protected_top_n] if protected_top_n and protected_top_n > 0 else []
    candidate_set = set(candidate_pool)
    prompt = apply_qwen_prefix(LLM_RAG_RANKED_SELECTOR_PROMPT_MIMIC, llm)
    chain = prompt | llm | StrOutputParser()
    try:
        raw_result = chain.invoke({
            "patient_context": patient_context,
            "recent_visit_history_text": _format_recent_visit_history_text(recent_visit_history, max_visits=3),
            "active_history": active_history if active_history else "None",
            "protected_medications": format_medication_list_for_llm(protected, atc_semantics),
            "ranked_candidates": _format_ranked_candidates_for_llm(candidate_pool, candidate_vote_scores, atc_semantics),
            "score_summary": _format_score_summary_for_llm(candidate_pool, candidate_vote_scores),
            "min_final_meds": min_final_meds,
            "max_final_meds": max_final_meds if max_final_meds and max_final_meds > 0 else len(candidate_pool),
        })
    except Exception as exc:
        fallback_n = min(19, len(candidate_pool))
        if max_final_meds and max_final_meds > 0:
            fallback_n = min(fallback_n, max_final_meds)
        fallback_n = max(min_final_meds if min_final_meds and min_final_meds > 0 else 1, fallback_n)
        fallback_n = min(fallback_n, len(candidate_pool))
        fallback = candidate_pool[:fallback_n]
        return {
            "final_prescription": _dedup_keep_order(fallback),
            "audit_log": [{
                "action": "KEPT",
                "drug": "rag_vote_fallback",
                "reason": f"Selector API call failed; RAG top-{fallback_n} fallback used: {exc}",
            }],
            "calibration_log": [],
            "final_description": "RAG vote fallback was used because selector API call failed.",
            "raw_output": "",
            "raw_model_output": "",
            "selection_strategy": "vote_llm_select",
            "estimated_target_size": fallback_n,
            "protected_high_vote_medications": protected,
            "rag_vote_top": candidate_pool,
            "parse_failed": True,
            "candidate_constraint": {
                "num_candidates": len(candidate_set),
                "removed_out_of_candidate": [],
            },
        }
    result = strip_think_tags(raw_result).strip()

    parse_failed = False
    removed_out_of_candidate = []
    calibration_log = []
    try:
        parsed = parse_llm_json_object(result)
        llm_meds = parsed.get("final_prescription", [])
        if isinstance(llm_meds, str):
            llm_meds = [s.strip() for s in llm_meds.split(",")]
        llm_meds = [str(med).strip() for med in llm_meds if str(med).strip()]
        constrained = [med for med in llm_meds if med in candidate_set]
        removed_out_of_candidate = [med for med in llm_meds if med not in candidate_set]
        calibration_log = parsed.get("calibration_log", [])
        if not isinstance(calibration_log, list):
            calibration_log = []
    except Exception as exc:
        parse_failed = True
        fallback_n = min_final_meds if min_final_meds and min_final_meds > 0 else min(10, len(candidate_pool))
        constrained = candidate_pool[:fallback_n]
        parsed = {"final_description": f"RAG vote fallback used because selector output could not be parsed: {exc}"}

    protected_added = [med for med in protected if med not in set(constrained)]
    if protected_added:
        calibration_log.append({
            "action": "KEPT",
            "drug": "protected_anchor",
            "reason": f"Re-added protected high-vote medications omitted by selector: {', '.join(protected_added)}",
        })
    final = _dedup_keep_order(protected + constrained)
    target_min = min_final_meds if min_final_meds and min_final_meds > 0 else min(8, len(candidate_pool))
    for med in candidate_pool:
        if len(final) >= target_min:
            break
        if med not in set(final):
            final.append(med)
    final = _dedup_keep_order(final)

    if max_final_meds and max_final_meds > 0:
        protected_kept = [med for med in protected if med in final]
        rest = [med for med in final if med not in set(protected_kept)]
        final = _dedup_keep_order(protected_kept + rest)[:max_final_meds]

    return {
        "final_prescription": final,
        "audit_log": calibration_log,
        "calibration_log": calibration_log,
        "final_description": parsed.get("final_description", ""),
        "raw_output": result,
        "raw_model_output": raw_result,
        "selection_strategy": "vote_llm_select",
        "estimated_target_size": parsed.get("estimated_target_size"),
        "protected_high_vote_medications": protected,
        "rag_vote_top": candidate_pool,
        "parse_failed": parse_failed,
        "candidate_constraint": {
            "num_candidates": len(candidate_set),
            "removed_out_of_candidate": removed_out_of_candidate,
            "protected_anchor_added": protected_added,
        },
    }


def call_vote_tail_pruner(
    diagnoses,
    active_history,
    recent_visit_history,
    llm,
    ranked_candidates,
    patient_context=None,
    protected_top_n=10,
    vote_top_n=15,
    max_remove=3,
    min_final_meds=18,
    atc_semantics=None,
):
    vote_pool = ranked_candidates[:vote_top_n] if vote_top_n and vote_top_n > 0 else ranked_candidates[:20]
    protected = vote_pool[:protected_top_n] if protected_top_n and protected_top_n > 0 else []
    tail = vote_pool[len(protected):]
    prompt = apply_qwen_prefix(LLM_VOTE_TAIL_PRUNER_PROMPT_MIMIC, llm)
    chain = prompt | llm | StrOutputParser()
    raw_result = chain.invoke({
        "patient_context": patient_context or f"Diagnoses: {', '.join(diagnoses) if diagnoses else 'None'}",
        "recent_visit_history_text": _format_recent_visit_history_text(recent_visit_history, max_visits=3),
        "protected_medications": format_medication_list_for_llm(protected, atc_semantics),
        "tail_candidates": format_medication_list_for_llm(tail, atc_semantics),
        "max_remove": max_remove,
        "min_final_meds": min_final_meds,
    })
    result = strip_think_tags(raw_result).strip()

    remove = []
    parse_failed = False
    try:
        parsed = parse_llm_json_object(result)
        remove = parsed.get("remove", [])
        if isinstance(remove, str):
            remove = [s.strip() for s in remove.split(",")]
        remove = [str(med).strip() for med in remove if str(med).strip()]
    except Exception:
        parsed = {"reason": "Tail pruner parse failed; no candidates removed."}
        parse_failed = True

    tail_set = set(tail)
    protected_set = set(protected)
    remove = [med for med in _dedup_keep_order(remove) if med in tail_set and med not in protected_set][:max_remove]
    if parse_failed:
        # If the calibrator does not return usable JSON, fall back to a bounded
        # RAG-vote list instead of the full tail. This avoids hurting precision.
        fallback_n = min_final_meds if min_final_meds and min_final_meds > 0 else len(protected)
        final = vote_pool[: max(len(protected), min(fallback_n, len(vote_pool)))]
    else:
        final = [med for med in vote_pool if med not in set(remove)]

    # If pruning made the list too short, fill from the next RAG-vote ranks.
    if min_final_meds and min_final_meds > 0 and len(final) < min_final_meds:
        for med in ranked_candidates[vote_top_n:]:
            if len(final) >= min_final_meds:
                break
            if med not in set(final):
                final.append(med)

    return {
        "final_prescription": _dedup_keep_order(final),
        "audit_log": [{
            "action": "REMOVED",
            "drug": med,
            "reason": "Removed by LLM tail-pruning from low-priority RAG candidates.",
        } for med in remove],
        "calibration_log": [{
            "action": "REMOVED",
            "drug": med,
            "reason": "Removed by LLM tail-pruning from low-priority RAG candidates.",
        } for med in remove],
        "final_description": parsed.get("reason", ""),
        "raw_output": result,
        "raw_model_output": raw_result,
        "selection_strategy": "vote_llm_prune",
        "protected_high_vote_medications": protected,
        "rag_vote_top": vote_pool,
        "tail_candidates": tail,
        "removed_by_tail_pruner": remove,
        "parse_failed": parse_failed,
        "candidate_constraint": {
            "num_candidates": len(vote_pool),
            "removed_out_of_candidate": [],
        },
    }


def update_summary(results: List[Dict]) -> Dict:
    f1s, precs, recs, jaccards = [], [], [], []
    init_f1s, init_precs, init_recs, init_jaccards = [], [], [], []
    llm_calls, runtimes = [], []
    candidate_recalls, candidate_sizes, final_sizes, gt_sizes = [], [], [], []

    for result in results:
        metrics = result.get("metrics", {})
        f1s.append(metrics.get("f1", 0.0))
        precs.append(metrics.get("precision", 0.0))
        recs.append(metrics.get("recall", 0.0))
        jaccards.append(metrics.get("jaccard", 0.0))
        init_f1s.append(metrics.get("initial_f1", 0.0))
        init_precs.append(metrics.get("initial_precision", 0.0))
        init_recs.append(metrics.get("initial_recall", 0.0))
        init_jaccards.append(metrics.get("initial_jaccard", 0.0))
        llm_calls.append(metrics.get("llm_calls", 0))
        runtimes.append(metrics.get("runtime_seconds", 0.0))
        candidate_recalls.append(metrics.get("candidate_recall_upper_bound", 0.0))
        candidate_sizes.append(metrics.get("candidate_size", 0))
        final_sizes.append(metrics.get("final_size", 0))
        gt_sizes.append(metrics.get("ground_truth_size", 0))

    return {
        "num_samples": len(results),
        "macro_f1": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "macro_precision": round(statistics.mean(precs), 4) if precs else 0.0,
        "macro_recall": round(statistics.mean(recs), 4) if recs else 0.0,
        "macro_jaccard": round(statistics.mean(jaccards), 4) if jaccards else 0.0,
        "initial_macro_f1": round(statistics.mean(init_f1s), 4) if init_f1s else 0.0,
        "initial_macro_precision": round(statistics.mean(init_precs), 4) if init_precs else 0.0,
        "initial_macro_recall": round(statistics.mean(init_recs), 4) if init_recs else 0.0,
        "initial_macro_jaccard": round(statistics.mean(init_jaccards), 4) if init_jaccards else 0.0,
        "avg_llm_calls_per_patient": round(statistics.mean(llm_calls), 4) if llm_calls else 0.0,
        "avg_runtime_seconds_per_patient": round(statistics.mean(runtimes), 4) if runtimes else 0.0,
        "avg_candidate_recall_upper_bound": round(statistics.mean(candidate_recalls), 4) if candidate_recalls else 0.0,
        "avg_candidate_size": round(statistics.mean(candidate_sizes), 4) if candidate_sizes else 0.0,
        "avg_final_size": round(statistics.mean(final_sizes), 4) if final_sizes else 0.0,
        "avg_ground_truth_size": round(statistics.mean(gt_sizes), 4) if gt_sizes else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Candidate-constrained multi-model PACE-RAG experiment")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vector_db", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--rag_index_name", type=str, default="milti-source")
    parser.add_argument(
        "--use_clinical_state_query",
        action="store_true",
        default=True,
        help="Use clinical_state/procedures/lab_summary fields as the retrieval query.",
    )
    parser.add_argument(
        "--diagnosis_only_query",
        action="store_true",
        help="Rich-context variant only: disable the default clinical_state query and retrieve with diagnosis/focus text.",
    )
    parser.add_argument(
        "--enable_field_aware_retrieval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FAISS high-recall retrieval followed by field-aware re-ranking with missing-field masks.",
    )
    parser.add_argument(
        "--field_aware_stage1_k",
        type=int,
        default=50,
        help="Number of FAISS candidates to retrieve before field-aware re-ranking.",
    )
    parser.add_argument(
        "--field_weights",
        default="",
        help=(
            "Comma-separated field weights, e.g. "
            "semantic=0.30,diagnoses=0.30,labs=0.16,vitals=0.08,demographics=0.03."
        ),
    )
    parser.add_argument("--llm_model", type=str, default=DEFAULT_OPENAI_MODEL, help="Backward-compatible default model.")
    parser.add_argument("--generator_model", type=str, default="", help="Focus/draft/RAG-tendency/summary model.")
    parser.add_argument("--verifier_model", type=str, default="", help="Candidate-constrained verifier model.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm_num_predict", type=int, default=512, help="Max output tokens for each LLM call.")
    parser.add_argument("--llm_json_mode", action="store_true", help="Use Ollama JSON mode for LLM calls that require JSON output.")
    parser.add_argument(
        "--ollama_num_ctx",
        type=int,
        default=8192,
        help="Ollama context window. Increase for local Qwen if prompts are truncated or failing.",
    )
    parser.add_argument(
        "--ollama_keep_alive",
        default="10m",
        help="How long Ollama should keep the model loaded, e.g. 10m, 30m, or -1.",
    )
    parser.add_argument(
        "--ollama_num_thread",
        type=int,
        default=0,
        help="Optional Ollama CPU thread count. 0 lets Ollama choose.",
    )
    parser.add_argument(
        "--ollama_reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable Ollama reasoning mode. For Qwen JSON selection, default is disabled.",
    )
    parser.add_argument(
        "--llm_provider",
        choices=["auto", "ollama", "openai", "messages"],
        default="openai",
        help="LLM backend. Use openai for /chat/completions and messages for /messages APIs.",
    )
    parser.add_argument(
        "--openai_base_url",
        default=DEFAULT_OPENAI_BASE_URL,
        help="OpenAI-compatible base URL, for example https://api.bitidea.cn/v1.",
    )
    parser.add_argument(
        "--openai_api_key_env",
        default=DEFAULT_OPENAI_API_KEY_ENV,
        help="Environment variable name that stores the OpenAI-compatible API key.",
    )
    parser.add_argument(
        "--openai_api_key",
        default="",
        help="OpenAI-compatible API key. Prefer setting the environment variable instead of passing this on the command line.",
    )
    parser.add_argument(
        "--llm_call_pause_seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds before each OpenAI-compatible API call to reduce provider-side connection drops.",
    )
    parser.add_argument(
        "--atc_semantics_csv",
        default="",
        help="Optional CSV with atc_code, atc_name, top_drug_names to make ATC codes understandable to LLM.",
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--retrieve_patients", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Randomly sample this many test rows with --sample_seed. Takes precedence over --max_samples.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed used by --sample_size.",
    )
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="Resume from an existing output JSON and skip patient_id values already saved.",
    )
    parser.add_argument(
        "--rerun_failed_existing",
        action="store_true",
        help="With --resume_existing, keep successful saved results but rerun records whose final_answer.parse_failed is true.",
    )
    parser.add_argument("--disable_doctor_summary", action="store_true")
    parser.add_argument("--disable_rag_tendency", action="store_true", help="Skip RAG tendency LLM calls for faster experiments.")
    parser.add_argument(
        "--include_draft_in_candidates",
        action="store_true",
        help="Ablation option: include initial draft medications in the candidate set.",
    )
    parser.add_argument(
        "--exclude_query_med_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove Medication history labels from clinical_state query/context text to avoid label leakage/noise.",
    )
    parser.add_argument(
        "--skip_focus_extraction",
        action="store_true",
        help="Skip LLM focus extraction. Useful when using clinical_state query and RAG tendency is disabled.",
    )
    parser.add_argument(
        "--selection_strategy",
        choices=["verifier", "rag_vote", "hybrid", "vote_llm", "vote_llm_prune", "vote_llm_select"],
        default="verifier",
        help="Final medication selection strategy. vote_llm_select uses GPT to choose a compact final set from RAG-ranked candidates.",
    )
    parser.add_argument(
        "--vote_top_n",
        type=int,
        default=10,
        help="Number of top ranked RAG-vote candidate medications to use for rag_vote/hybrid.",
    )
    parser.add_argument(
        "--min_final_meds",
        type=int,
        default=0,
        help="For hybrid strategy, fill from RAG votes until at least this many final medications are present.",
    )
    parser.add_argument(
        "--max_final_meds",
        type=int,
        default=0,
        help="Optional cap on final prescription size for rag_vote/hybrid.",
    )
    parser.add_argument(
        "--llm_protect_top_n",
        type=int,
        default=5,
        help="For vote_llm, force-protect this many highest RAG-vote medications from aggressive LLM deletion.",
    )
    parser.add_argument(
        "--llm_max_remove",
        type=int,
        default=3,
        help="For vote_llm_prune, allow LLM to remove at most this many tail candidates.",
    )
    parser.add_argument(
        "--enable_size_guard",
        action="store_true",
        help="For vote_llm_select, conservatively trim obvious small-prescription over-selection and fill obvious large-prescription under-selection.",
    )
    parser.add_argument(
        "--enable_size_calibration",
        action="store_true",
        help="Backward-compatible alias for --enable_size_guard.",
    )
    parser.add_argument(
        "--size_calibration_slack",
        type=int,
        default=2,
        help="Deprecated compatibility option. The conservative size guard does not use this value.",
    )
    args = parser.parse_args()
    if args.diagnosis_only_query:
        args.use_clinical_state_query = False
    field_weights = parse_field_weights(args.field_weights)

    generator_model = args.generator_model or args.llm_model
    verifier_model = args.verifier_model or args.llm_model
    draft_is_skipped = (
        args.selection_strategy in {"rag_vote", "vote_llm", "vote_llm_prune", "vote_llm_select"}
        and not args.include_draft_in_candidates
    )
    generator_needed = (
        not args.skip_focus_extraction
        or not args.disable_rag_tendency
        or not draft_is_skipped
        or not args.disable_doctor_summary
    )
    verifier_needed = args.selection_strategy in {
        "verifier",
        "hybrid",
        "vote_llm",
        "vote_llm_prune",
        "vote_llm_select",
    }
    generator_llm = (
        make_llm(
            generator_model,
            args.temperature,
            args.seed,
            args.llm_num_predict,
            args.llm_json_mode,
            args.llm_provider,
            args.openai_base_url,
            args.openai_api_key_env,
            args.openai_api_key,
            args.llm_call_pause_seconds,
            args.ollama_num_ctx,
            args.ollama_keep_alive,
            args.ollama_num_thread,
            args.ollama_reasoning,
        )
        if generator_needed
        else None
    )
    verifier_llm = (
        make_llm(
            verifier_model,
            args.temperature,
            args.seed,
            args.llm_num_predict,
            args.llm_json_mode,
            args.llm_provider,
            args.openai_base_url,
            args.openai_api_key_env,
            args.openai_api_key,
            args.llm_call_pause_seconds,
            args.ollama_num_ctx,
            args.ollama_keep_alive,
            args.ollama_num_thread,
            args.ollama_reasoning,
        )
        if verifier_needed
        else None
    )
    atc_semantics = load_atc_semantics(args.atc_semantics_csv)

    df = pd.read_csv(args.data_path)
    subject_groups = {
        sid: group.sort_values("admittime").to_dict("records")
        for sid, group in df.groupby("subject_id")
    }
    vs = FAISS.load_local(
        args.vector_db,
        embedding_model,
        index_name=args.rag_index_name,
        allow_dangerous_deserialization=True,
    )

    if args.sample_size and args.sample_size > 0:
        sample_n = min(args.sample_size, len(df))
        df = df.sample(n=sample_n, random_state=args.sample_seed).reset_index(drop=True)
        print(f"[INFO] Randomly sampled {sample_n} rows with seed={args.sample_seed}")
    elif args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    results = []
    if args.resume_existing and Path(args.output_file).exists():
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            existing_results = existing_payload.get("results", [])
            if isinstance(existing_results, list):
                if args.rerun_failed_existing:
                    results = [
                        item for item in existing_results
                        if not (isinstance(item, dict) and (item.get("final_answer") or {}).get("parse_failed"))
                    ]
                    print(
                        f"[INFO] Resuming from existing output: kept {len(results)} successful results; "
                        f"will rerun {len(existing_results) - len(results)} failed results"
                    )
                else:
                    results = existing_results
                    print(f"[INFO] Resuming from existing output: {len(results)} saved results")
        except Exception as exc:
            print(f"[WARN] Could not resume from {args.output_file}: {exc}")
    processed_patient_keys = {
        str(item.get("patient_id", "")).strip()
        for item in results
        if isinstance(item, dict) and str(item.get("patient_id", "")).strip()
    }
    summary = update_summary(results)
    total_patients = len(df)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=total_patients, desc="Processing Candidate Multi-Model RAG"), start=1):
        patient_start = time.perf_counter()
        llm_call_count = 0
        subject_id, hadm_id = row["subject_id"], row["hadm_id"]
        patient_key = f"{subject_id}_{hadm_id}"
        if patient_key in processed_patient_keys:
            continue
        stage_logs = []

        def log_stage(stage_name: str, detail: str = ""):
            msg = f"[STAGE][{idx}/{total_patients}][{patient_key}] {stage_name}"
            if detail:
                msg += f" | {detail}"
            print(msg, flush=True)
            stage_logs.append(msg)

        log_stage("START", f"generator={generator_model}, verifier={verifier_model}")
        diagnoses = parse_list_cell(row["diagnoses"])
        ground_truth_meds = parse_list_cell(row["medications"])
        patient_context_for_llm = build_patient_context_for_llm(
            row,
            diagnoses,
            include_medication_history=not args.exclude_query_med_history,
        )

        history = subject_groups[subject_id]
        curr_idx = next(i for i, record in enumerate(history) if record["hadm_id"] == hadm_id)
        prev_visits = history[:curr_idx]
        active_history_list = parse_list_cell(prev_visits[-1]["medications"]) if prev_visits else []
        recent_visit_history = []
        for visit in prev_visits[-3:]:
            recent_visit_history.append({
                "visit": f"Visit ({visit['admittime']})",
                "symptoms": ", ".join(parse_list_cell(visit["diagnoses"])),
                "prescription": parse_list_cell(visit["medications"]),
            })
        log_stage("HISTORY_READY", f"active_history={len(active_history_list)}, recent_visits={len(recent_visit_history)}")

        if args.skip_focus_extraction:
            focus_queries = [", ".join(diagnoses)] if diagnoses else []
            log_stage("FOCUS_EXTRACT_SKIPPED", f"fallback_focus_keywords={len(focus_queries)}")
        else:
            focus_queries = LLM_extract_focus_keywords_MIMIC(diagnoses, generator_llm)
            llm_call_count += 1
            log_stage("FOCUS_EXTRACT_DONE", f"focus_keywords={len(focus_queries)}")

        if args.use_clinical_state_query:
            search_queries = [build_clinical_state_query(row, diagnoses, include_medication_history=not args.exclude_query_med_history)]
        else:
            search_queries = [query for query in focus_queries if query] or [", ".join(diagnoses)]
        similar_docs_map = {}
        query_level_hits = []
        for query in search_queries:
            stage1_k = args.retrieve_patients
            if args.enable_field_aware_retrieval:
                stage1_k = max(args.retrieve_patients, args.field_aware_stage1_k)
            hits = vs.similarity_search_with_relevance_scores(query, k=stage1_k)
            query_hits = []
            for doc, semantic_score in hits:
                semantic_score_f = safe_float(semantic_score)
                hit = build_rag_hit_payload(doc, semantic_score_f)
                final_score = semantic_score_f
                field_scores = {}
                if args.enable_field_aware_retrieval:
                    final_score, field_scores = field_aware_score(row, hit, semantic_score_f, field_weights)
                    hit["semantic_score"] = semantic_score_f
                    hit["field_aware_score"] = safe_float(final_score)
                    hit["field_scores"] = field_scores
                    hit["score"] = safe_float(final_score)
                if final_score >= args.threshold:
                    query_hits.append(hit)
                    key = (hit.get("source"), hit.get("subject_id"), hit.get("hadm_id"))
                    if key not in similar_docs_map or final_score > safe_float(similar_docs_map[key].get("score", 0.0)):
                        similar_docs_map[key] = hit
            query_level_hits.append({
                "query": query,
                "topk": args.retrieve_patients,
                "stage1_k": stage1_k,
                "threshold": args.threshold,
                "retrieval_mode": "field_aware_rerank" if args.enable_field_aware_retrieval else "faiss_only",
                "hits": query_hits,
            })
        log_stage("RAG_RETRIEVE_DONE", f"queries={len(search_queries)}, passed_hits={sum(len(q.get('hits', [])) for q in query_level_hits)}")

        rag_patients = []
        for hit in sorted(similar_docs_map.values(), key=lambda value: safe_float(value.get("score", 0.0)), reverse=True)[:args.retrieve_patients]:
            rag_patients.append(hit)
        rag_source_counts = count_sources(rag_patients)
        log_stage("RAG_SELECT_DONE", f"selected_similar_patients={len(rag_patients)}, sources={rag_source_counts}")

        rag_tendency_by_focus = []
        if args.disable_rag_tendency:
            log_stage("RAG_TENDENCY_SKIPPED", "disabled by flag")
        else:
            tendency_targets = []
            if args.use_clinical_state_query:
                for item in query_level_hits:
                    query_text = str(item.get("query", "") or "")
                    tendency_targets.append({
                        "focus": "clinical_state_context",
                        "source": "clinical_state_query",
                        "hits": item.get("hits", []),
                        "diagnoses_for_prompt": focus_queries if focus_queries else diagnoses,
                        "query_text": query_text,
                    })
            else:
                focus_set = set(focus_queries or [])
                for item in query_level_hits:
                    query_text = str(item.get("query", "") or "")
                    tendency_targets.append({
                        "focus": query_text or "diagnosis_context",
                        "source": "focus_keyword" if query_text in focus_set else "diagnosis_fallback",
                        "hits": item.get("hits", []),
                        "diagnoses_for_prompt": [query_text] if query_text else diagnoses,
                        "query_text": query_text,
                    })

            for target in tendency_targets:
                focus = target["focus"]
                focus_hits = target.get("hits", [])
                if not focus_hits:
                    continue
                focus_cases = [{
                    "content": build_retrieved_case_context(hit),
                    "medications": hit.get("medications", []),
                    "score": hit.get("score", 0.0),
                    "source": hit.get("source", ""),
                    "subject_id": hit.get("subject_id"),
                    "hadm_id": hit.get("hadm_id"),
                    "diagnoses": hit.get("diagnoses", []),
                    "procedures": hit.get("procedures", []),
                    "lab_summary": hit.get("lab_summary", []),
                    "vital_summary": hit.get("vital_summary", []),
                    "nurse_charting_summary": hit.get("nurse_charting_summary", []),
                } for hit in focus_hits]
                if not focus_cases:
                    continue
                focus_tendency = call_LLM_rag_tendency_analyzer_MIMIC(
                    rag_patients=focus_cases,
                    diagnoses=target.get("diagnoses_for_prompt") or [focus],
                    llm=generator_llm,
                )
                llm_call_count += 1
                rag_tendency_by_focus.append({
                    "focus": focus,
                    "source": target.get("source", ""),
                    "query": target.get("query_text", ""),
                    "num_cases": len(focus_cases),
                    "source_counts": count_sources(focus_cases),
                    "tendency": focus_tendency,
                })
            log_stage("RAG_TENDENCY_DONE", f"tendency_items={len(rag_tendency_by_focus)}")

        if args.selection_strategy in {"rag_vote", "vote_llm", "vote_llm_prune", "vote_llm_select"} and not args.include_draft_in_candidates:
            initial_prescription_raw = "[START]\n[END]"
            log_stage("INITIAL_DRAFT_SKIPPED", f"{args.selection_strategy} without draft candidates")
        else:
            initial_prescription_raw = call_LLM_simple_prescription_with_reason_prompt_MIMIC(
                diagnoses,
                active_history_list,
                generator_llm,
                recent_visit_history=recent_visit_history,
            )
            llm_call_count += 1
            log_stage("INITIAL_DRAFT_DONE", "initial draft generated")

        candidate_medications = build_candidate_medications(
            active_history=active_history_list,
            rag_patients=rag_patients,
            rag_tendency_by_focus=rag_tendency_by_focus,
            initial_prescription=initial_prescription_raw,
            include_initial_draft=args.include_draft_in_candidates,
        )
        ranked_candidates, candidate_vote_scores = rank_candidate_medications(
            active_history=active_history_list,
            rag_patients=rag_patients,
            rag_tendency_by_focus=rag_tendency_by_focus,
            initial_prescription=initial_prescription_raw,
            include_initial_draft=args.include_draft_in_candidates,
        )
        log_stage(
            "CANDIDATE_SET_DONE",
            f"candidate_medications={len(candidate_medications)}, ranked_candidates={len(ranked_candidates)}",
        )

        if args.selection_strategy == "rag_vote":
            final_result = {
                "final_prescription": [],
                "audit_log": [],
                "final_description": "RAG vote selection was used; verifier skipped.",
                "raw_output": "",
                "candidate_constraint": {
                    "num_candidates": len(candidate_medications),
                    "removed_out_of_candidate": [],
                },
            }
            log_stage("CANDIDATE_VERIFIER_SKIPPED", "rag_vote strategy")
        elif args.selection_strategy == "vote_llm":
            vote_top_n = args.vote_top_n if args.vote_top_n and args.vote_top_n > 0 else 12
            min_final_meds = args.min_final_meds if args.min_final_meds and args.min_final_meds > 0 else min(8, vote_top_n)
            max_final_meds = args.max_final_meds if args.max_final_meds and args.max_final_meds > 0 else vote_top_n
            final_result = call_vote_guided_calibrator(
                diagnoses=diagnoses,
                active_history=active_history_list,
                recent_visit_history=recent_visit_history,
                llm=verifier_llm,
                ranked_candidates=ranked_candidates,
                candidate_vote_scores=candidate_vote_scores,
                patient_context=patient_context_for_llm,
                protected_top_n=args.llm_protect_top_n,
                vote_top_n=vote_top_n,
                min_final_meds=min_final_meds,
                max_final_meds=max_final_meds,
                atc_semantics=atc_semantics,
            )
            llm_call_count += 1
            log_stage("VOTE_GUIDED_CALIBRATOR_DONE", f"vote_top_n={vote_top_n}, protected_top_n={args.llm_protect_top_n}")
        elif args.selection_strategy == "vote_llm_select":
            vote_top_n = args.vote_top_n if args.vote_top_n and args.vote_top_n > 0 else 20
            min_final_meds = args.min_final_meds if args.min_final_meds and args.min_final_meds > 0 else min(8, vote_top_n)
            max_final_meds = args.max_final_meds if args.max_final_meds and args.max_final_meds > 0 else min(12, vote_top_n)
            final_result = call_rag_ranked_selector(
                diagnoses=diagnoses,
                active_history=active_history_list,
                recent_visit_history=recent_visit_history,
                llm=verifier_llm,
                ranked_candidates=ranked_candidates,
                candidate_vote_scores=candidate_vote_scores,
                patient_context=patient_context_for_llm,
                protected_top_n=args.llm_protect_top_n,
                vote_top_n=vote_top_n,
                min_final_meds=min_final_meds,
                max_final_meds=max_final_meds,
                atc_semantics=atc_semantics,
            )
            if args.enable_size_guard or args.enable_size_calibration:
                final_result = apply_size_guard_postprocess(
                    final_result=final_result,
                    ranked_candidates=ranked_candidates,
                    candidate_vote_scores=candidate_vote_scores,
                    active_history=active_history_list,
                    rag_patients=rag_patients,
                    vote_top_n=vote_top_n,
                    max_final_meds=max_final_meds,
                )
            llm_call_count += 1
            log_stage(
                "VOTE_RANKED_SELECTOR_DONE",
                f"vote_top_n={vote_top_n}, protected_top_n={args.llm_protect_top_n}, min_final_meds={min_final_meds}, max_final_meds={max_final_meds}, size_guard={(args.enable_size_guard or args.enable_size_calibration)}",
            )
        elif args.selection_strategy == "vote_llm_prune":
            vote_top_n = args.vote_top_n if args.vote_top_n and args.vote_top_n > 0 else 20
            min_final_meds = args.min_final_meds if args.min_final_meds and args.min_final_meds > 0 else max(1, vote_top_n - args.llm_max_remove)
            final_result = call_vote_tail_pruner(
                diagnoses=diagnoses,
                active_history=active_history_list,
                recent_visit_history=recent_visit_history,
                llm=verifier_llm,
                ranked_candidates=ranked_candidates,
                patient_context=patient_context_for_llm,
                protected_top_n=args.llm_protect_top_n,
                vote_top_n=vote_top_n,
                max_remove=args.llm_max_remove,
                min_final_meds=min_final_meds,
                atc_semantics=atc_semantics,
            )
            llm_call_count += 1
            log_stage(
                "VOTE_TAIL_PRUNER_DONE",
                f"vote_top_n={vote_top_n}, protected_top_n={args.llm_protect_top_n}, max_remove={args.llm_max_remove}",
            )
        else:
            final_result = call_candidate_delta_verifier(
                initial_prescription=initial_prescription_raw,
                diagnoses=diagnoses,
                active_history=active_history_list,
                llm=verifier_llm,
                candidate_medications=candidate_medications,
                rag_tendency_by_focus=rag_tendency_by_focus,
                recent_visit_history=recent_visit_history,
                patient_context=patient_context_for_llm,
            )
            llm_call_count += 1
            log_stage("CANDIDATE_VERIFIER_DONE", "final recommendation generated")
        final_result = apply_final_strategy(
            final_result,
            ranked_candidates=ranked_candidates,
            strategy=args.selection_strategy,
            vote_top_n=args.vote_top_n,
            min_final_meds=args.min_final_meds,
            max_final_meds=args.max_final_meds,
        )
        log_stage(
            "FINAL_SELECTION_DONE",
            f"strategy={args.selection_strategy}, final_meds={len(final_result.get('final_prescription', [])) if isinstance(final_result, dict) else 0}",
        )

        doctor_summary = ""
        if not args.disable_doctor_summary:
            patient_state = f"S: Patient admitted on {row.get('admittime', 'N/A')} | O: Diagnoses: {diagnoses} | A: Evaluation for {diagnoses}"
            doctor_summary = call_LLM_doctor_summary(
                patient_state=patient_state,
                initial_prescription=initial_prescription_raw,
                rag_tendency_by_focus=rag_tendency_by_focus,
                audit_log=final_result.get("audit_log", []) if isinstance(final_result, dict) else [],
                final_answer_list=_final_payload(None, final_result),
                llm=generator_llm,
            )
            llm_call_count += 1
            log_stage("DOCTOR_SUMMARY_DONE", "doctor summary generated")
        else:
            log_stage("DOCTOR_SUMMARY_SKIPPED", "disabled by flag")

        init_metrics = list_medicine_answer(initial_prescription_raw, ground_truth_meds)
        final_metrics = list_medicine_answer(final_result, ground_truth_meds)
        if1, iprec, irec = compute_scores(init_metrics)
        f1, prec, rec = compute_scores(final_metrics)
        initial_jaccard = compute_jaccard(initial_prescription_raw, ground_truth_meds)
        final_jaccard = compute_jaccard(final_result, ground_truth_meds)
        candidate_stats = set_overlap_stats(candidate_medications, ground_truth_meds)
        final_stats = set_overlap_stats(
            final_result.get("final_prescription", []) if isinstance(final_result, dict) else [],
            ground_truth_meds,
        )
        runtime_seconds = time.perf_counter() - patient_start

        patient_result = {
            "patient_id": patient_key,
            "diagnoses": diagnoses,
            "ground_truth": ground_truth_meds,
            "ground_truth_list": ground_truth_meds,
            "active_history": active_history_list,
            "recent_visit_history": recent_visit_history,
            "rag_patients": rag_patients if rag_patients else None,
            "rag_tendency_by_focus": rag_tendency_by_focus,
            "candidate_medications": candidate_medications,
            "model_config": {
                "generator_model": generator_model,
                "verifier_model": verifier_model,
                "doctor_summary_enabled": not args.disable_doctor_summary,
                "rag_tendency_enabled": not args.disable_rag_tendency,
                "rag_index_name": args.rag_index_name,
                "use_clinical_state_query": args.use_clinical_state_query,
                "exclude_query_med_history": args.exclude_query_med_history,
                "skip_focus_extraction": args.skip_focus_extraction,
                "diagnosis_only_query": args.diagnosis_only_query,
                "selection_strategy": args.selection_strategy,
                "enable_field_aware_retrieval": args.enable_field_aware_retrieval,
                "field_aware_stage1_k": args.field_aware_stage1_k,
                "field_weights": field_weights,
                "vote_top_n": args.vote_top_n,
                "min_final_meds": args.min_final_meds,
                "max_final_meds": args.max_final_meds,
                "llm_protect_top_n": args.llm_protect_top_n,
                "llm_max_remove": args.llm_max_remove,
                "enable_size_guard": args.enable_size_guard,
                "enable_size_calibration": args.enable_size_calibration,
                "effective_size_guard": bool(args.enable_size_guard or args.enable_size_calibration),
                "size_calibration_slack": args.size_calibration_slack,
                "atc_semantics_csv": args.atc_semantics_csv,
            },
            "rag_logging": {
                "focus_keywords": focus_queries,
                "search_queries": search_queries,
                "topk": args.retrieve_patients,
                "threshold": args.threshold,
                "retrieval_mode": "field_aware_rerank" if args.enable_field_aware_retrieval else "faiss_only",
                "field_weights": field_weights,
                "query_level_hits": query_level_hits,
                "selected_similar_patients": rag_patients,
                "selected_source_counts": rag_source_counts,
                "ranked_candidate_medications": ranked_candidates,
                "candidate_vote_scores": candidate_vote_scores,
            },
            "rich_patient_context": patient_context_for_llm,
            "final_llm_input_summary": {
                "context_chars": len(patient_context_for_llm),
                "has_procedures": bool(parse_list_cell(row.get("procedures")) if "procedures" in row else []),
                "has_lab_summary": bool(parse_list_cell(row.get("lab_summary")) if "lab_summary" in row else []),
                "has_vital_summary": bool(parse_list_cell(row.get("vital_summary")) if "vital_summary" in row else []),
                "has_nurse_charting_summary": bool(parse_list_cell(row.get("nurse_charting_summary")) if "nurse_charting_summary" in row else []),
                "rag_source_counts": rag_source_counts,
            },
            "stage_logs": stage_logs,
            "draft_plan": initial_prescription_raw,
            "final_answer": final_result,
            "doctor_summary": doctor_summary,
            "initial_answer_list": init_metrics,
            "final_answer_list": final_metrics,
            "metrics": {
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "jaccard": final_jaccard,
                "initial_f1": if1,
                "initial_precision": iprec,
                "initial_recall": irec,
                "initial_jaccard": initial_jaccard,
                "llm_calls": llm_call_count,
                "runtime_seconds": round(runtime_seconds, 4),
                "candidate_recall_upper_bound": candidate_stats["recall"],
                "candidate_precision_against_ground_truth": candidate_stats["precision"],
                "candidate_size": candidate_stats["pred_size"],
                "final_size": final_stats["pred_size"],
                "ground_truth_size": final_stats["gt_size"],
                "final_hit_size": final_stats["hit_size"],
            },
        }
        results.append(patient_result)
        summary = update_summary(results)
        log_stage(
            "METRICS_DONE",
            f"f1={f1:.4f}, jaccard={final_jaccard:.4f}, llm_calls={llm_call_count}, runtime={runtime_seconds:.2f}s",
        )
        with open(args.output_file, "w", encoding="utf-8") as file:
            json.dump({"summary": summary, "results": results}, file, indent=2, ensure_ascii=False)
        log_stage("SAVE_DONE", f"saved_to={args.output_file}")
        print(
            f"\n>>> Current Summary (n={len(results)}): "
            f"F1={summary['macro_f1']}, Precision={summary['macro_precision']}, "
            f"Recall={summary['macro_recall']}, Jaccard={summary['macro_jaccard']}, "
            f"AvgCalls={summary['avg_llm_calls_per_patient']}, "
            f"AvgTime={summary['avg_runtime_seconds_per_patient']}s"
        )

    if not results:
        with open(args.output_file, "w", encoding="utf-8") as file:
            json.dump({"summary": summary, "results": results}, file, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
