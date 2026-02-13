"""Feature extraction for doctor-patient matching experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


class FeatureExtractor:
    """Extract all features for doctor-patient matching."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Use sentence-transformers for embeddings.
        MiniLM outputs 384-dim embeddings (efficient for Kaggle).
        """
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(embedding_model)
        self.embed_dim = 384
        self._doctor_topic_embeddings: np.ndarray | None = None
        self._doctor_id_to_idx: Dict[str, int] = {}

        self.related_specialty_map = self._build_related_specialties()
        self.hospital_score_map = {
            "AIIMS Network": 1.00,
            "Apollo Hospitals": 0.95,
            "Medanta": 0.93,
            "Fortis Healthcare": 0.91,
            "Manipal Hospitals": 0.89,
            "Narayana Health": 0.88,
            "Max Healthcare": 0.87,
            "Aster Medcity": 0.86,
            "CARE Hospitals": 0.84,
            "KIMS Hospitals": 0.82,
        }
        self._keyword_stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "have",
            "been",
            "patient",
            "reports",
            "reporting",
            "clinical",
            "care",
            "doctor",
            "evaluation",
            "need",
        }

    @staticmethod
    def _build_related_specialties() -> Dict[str, set]:
        groups = [
            ["General Medicine", "Family Medicine", "Preventive Medicine", "Geriatrics"],
            ["Cardiology", "Cardiac Surgery", "Vascular Surgery", "Critical Care"],
            ["Neurology", "Neurosurgery", "Physical Medicine & Rehabilitation", "Pain Medicine"],
            ["Orthopedics", "Sports Medicine", "Physical Medicine & Rehabilitation", "Pain Medicine"],
            ["Pulmonology", "Sleep Medicine", "Critical Care", "Allergy & Immunology"],
            ["Gastroenterology", "Hepatology", "General Surgery"],
            ["Oncology", "Hematology", "Palliative Care"],
            ["Pediatrics", "Neonatology"],
            ["Infectious Disease", "Tropical Medicine", "General Medicine"],
            ["Gynecology", "General Surgery"],
            ["ENT", "Ophthalmology", "Allergy & Immunology"],
            ["Urology", "Nephrology"],
        ]
        mapping: Dict[str, set] = {}
        for group in groups:
            for spec in group:
                mapping.setdefault(spec, set()).update([x for x in group if x != spec])
        return mapping

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(r * c)

    def _tokenize_keywords(self, text: str) -> set:
        text = (text or "").lower()
        chars = [ch if ch.isalnum() or ch.isspace() else " " for ch in text]
        tokens = "".join(chars).split()
        return {tok for tok in tokens if len(tok) > 2 and tok not in self._keyword_stopwords}

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 0.0
        inter = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b)) + 1e-8
        return float(inter / union)

    def compute_embeddings(self, doctors_df: pd.DataFrame, cases_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for all doctors and cases.

        For doctors: encode expertise_description
        For cases: encode symptom_description

        Returns:
        - doctor_embeddings: (n_doctors, 384)
        - case_embeddings: (n_cases, 384)
        """
        doctor_texts = doctors_df["expertise_description"].fillna("").astype(str).tolist()
        case_texts = cases_df["symptom_description"].fillna("").astype(str).tolist()

        doctor_embeddings_np = self.encoder.encode(
            doctor_texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        case_embeddings_np = self.encoder.encode(
            case_texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        topic_texts = doctors_df["research_topics"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else str(x or "")
        )
        topic_embeddings_np = self.encoder.encode(
            topic_texts.tolist(),
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        self._doctor_topic_embeddings = topic_embeddings_np
        self._doctor_id_to_idx = {str(doc_id): idx for idx, doc_id in enumerate(doctors_df["doctor_id"].tolist())}

        return torch.from_numpy(doctor_embeddings_np), torch.from_numpy(case_embeddings_np)

    def compute_clinical_features(
        self,
        case: pd.Series,
        doctor: pd.Series,
        case_emb: torch.Tensor | np.ndarray,
        doctor_emb: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Clinical dimension features (4-dim):
        1. cosine_similarity: cos(case_emb, doctor_emb)
        2. specialty_match: 1.0 exact, 0.5 related, 0.0 wrong
        3. subspecialty_match: 1.0 if match, else 0.0
        4. keyword_overlap: Jaccard similarity of medical terms

        Returns: tensor (4,)
        """
        case_vec = self._to_numpy(case_emb)
        doctor_vec = self._to_numpy(doctor_emb)
        cosine_similarity = self._cosine(case_vec, doctor_vec)

        case_specialty = str(case.get("target_specialty", ""))
        doctor_specialty = str(doctor.get("specialty", ""))
        if doctor_specialty == case_specialty:
            specialty_match = 1.0
        elif doctor_specialty in self.related_specialty_map.get(case_specialty, set()):
            specialty_match = 0.5
        else:
            specialty_match = 0.0

        subspecialties = doctor.get("subspecialties", [])
        if not isinstance(subspecialties, list):
            subspecialties = []
        symptom_tokens = self._tokenize_keywords(str(case.get("symptom_description", "")))
        subspecialty_match = 1.0 if any(self._tokenize_keywords(ss).intersection(symptom_tokens) for ss in subspecialties) else 0.0

        doctor_text = f"{doctor.get('expertise_description', '')} {' '.join(subspecialties)}"
        keyword_overlap = self._jaccard(
            self._tokenize_keywords(str(case.get("symptom_description", ""))),
            self._tokenize_keywords(doctor_text),
        )

        return torch.tensor(
            [cosine_similarity, specialty_match, subspecialty_match, keyword_overlap],
            dtype=torch.float32,
        )

    def compute_pastwork_features(
        self,
        case: pd.Series,
        doctor: pd.Series,
        case_emb: torch.Tensor | np.ndarray,
        doctor_emb: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Past work dimension features (5-dim):
        1. publication_impact: normalized log(pubs * citations)
        2. topic_relevance: cos(case_emb, research_topics_emb)
        3. experience_score: min(years/20, 1.0)
        4. platform_performance: completion_rate
        5. external_reputation: hospital affiliation score

        Returns: tensor (5,)
        """
        pubs = float(doctor.get("publications_count", 0.0))
        review_score = float(doctor.get("review_score", 4.0))
        citations_proxy = max(1.0, pubs * (review_score * 5.0))
        publication_impact = float(np.log1p(pubs * citations_proxy) / np.log1p(100.0 * 2500.0))
        publication_impact = float(np.clip(publication_impact, 0.0, 1.0))

        case_vec = self._to_numpy(case_emb)
        topic_relevance = 0.0
        doc_id = str(doctor.get("doctor_id", ""))
        if self._doctor_topic_embeddings is not None and doc_id in self._doctor_id_to_idx:
            topic_vec = self._doctor_topic_embeddings[self._doctor_id_to_idx[doc_id]]
            topic_relevance = self._cosine(case_vec, topic_vec)
            topic_relevance = float(np.clip((topic_relevance + 1.0) / 2.0, 0.0, 1.0))

        years = float(doctor.get("years_experience", 0.0))
        experience_score = float(np.clip(years / 20.0, 0.0, 1.0))

        platform_performance = float(np.clip(doctor.get("consultation_completion_rate", 0.0), 0.0, 1.0))

        hospitals = doctor.get("hospital_affiliations", [])
        if not isinstance(hospitals, list):
            hospitals = [hospitals] if hospitals else []
        if hospitals:
            values = [self.hospital_score_map.get(str(h), 0.75) for h in hospitals]
            external_reputation = float(np.clip(np.mean(values), 0.0, 1.0))
        else:
            external_reputation = 0.7

        return torch.tensor(
            [publication_impact, topic_relevance, experience_score, platform_performance, external_reputation],
            dtype=torch.float32,
        )

    def compute_logistics_features(self, case: pd.Series, doctor: pd.Series) -> torch.Tensor:
        """
        Logistics dimension features (5-dim):
        1. availability_score: next slot availability
        2. language_match: 1.0 if preferred language available
        3. proximity_score: 1 - min(distance_km/100, 1)
        4. fee_match: 1.0 if in budget, linear decay outside
        5. mode_match: 1.0 if preference matches

        Returns: tensor (5,)
        """
        availability_score = float(np.clip(doctor.get("availability_score", 0.0), 0.0, 1.0))

        preferred_language = str(case.get("preferred_language", ""))
        languages = doctor.get("languages", [])
        if not isinstance(languages, list):
            languages = [languages] if languages else []
        language_match = 1.0 if preferred_language in languages else 0.0

        case_loc = case.get("location", {}) or {}
        doctor_loc = doctor.get("location", {}) or {}
        distance_km = self._haversine_km(
            float(case_loc.get("lat", 0.0)),
            float(case_loc.get("lon", 0.0)),
            float(doctor_loc.get("lat", 0.0)),
            float(doctor_loc.get("lon", 0.0)),
        )
        proximity_score = float(np.clip(1.0 - min(distance_km / 100.0, 1.0), 0.0, 1.0))

        budget_min, budget_max = case.get("budget_range", [0.0, 0.0])
        budget_min = float(budget_min)
        budget_max = float(budget_max)
        fee = float(doctor.get("consultation_fee", 0.0))
        if budget_min <= fee <= budget_max:
            fee_match = 1.0
        else:
            delta = min(abs(fee - budget_min), abs(fee - budget_max))
            fee_match = float(np.clip(1.0 - (delta / 3000.0), 0.0, 1.0))

        preferred_mode = str(case.get("preferred_mode", "no-preference"))
        available_mode = str(doctor.get("available_modes", "both"))
        mode_match = 1.0 if preferred_mode == "no-preference" or available_mode in {"both", preferred_mode} else 0.0

        return torch.tensor(
            [availability_score, language_match, proximity_score, fee_match, mode_match],
            dtype=torch.float32,
        )

    def compute_trust_features(self, doctor: pd.Series) -> torch.Tensor:
        """
        Trust dimension features (3-dim):
        1. nmc_verified: 1.0 or 0.0
        2. profile_completeness: 0.3-1.0
        3. review_score: normalized to 0-1

        Returns: tensor (3,)
        """
        nmc_verified = 1.0 if bool(doctor.get("nmc_verified", False)) else 0.0
        profile_completeness = float(np.clip(doctor.get("profile_completeness", 0.0), 0.0, 1.0))
        review_score = float(doctor.get("review_score", 2.5))
        review_norm = float(np.clip((review_score - 2.5) / (5.0 - 2.5), 0.0, 1.0))
        return torch.tensor([nmc_verified, profile_completeness, review_norm], dtype=torch.float32)

    def compute_context_features(self, case: pd.Series) -> torch.Tensor:
        """
        Context features for gating (8-dim):
        1. urgency_level: 0-3 normalized to 0-1
        2. symptom_count: normalized
        3. red_flag_score: 0-1
        4. patient_age: age/90
        5. comorbidity_count: count/5
        6. disease_rarity_score: 0-1
        7. is_pediatric: 1.0 if age < 18
        8. is_emergency: 1.0 if urgency = emergency

        Returns: tensor (8,)
        """
        urgency_map = {"routine": 0.0, "semi-urgent": 1.0 / 3.0, "urgent": 2.0 / 3.0, "emergency": 1.0}
        urgency = str(case.get("urgency_level", "routine"))
        urgency_level = float(urgency_map.get(urgency, 0.0))

        symptoms = case.get("symptoms", [])
        if not isinstance(symptoms, list):
            symptoms = [symptoms] if symptoms else []
        symptom_count = float(np.clip(len(symptoms) / 8.0, 0.0, 1.0))

        red_flag_score = float(np.clip(case.get("red_flag_score", 0.0), 0.0, 1.0))
        age = float(case.get("patient_age", 0.0))
        patient_age = float(np.clip(age / 90.0, 0.0, 1.0))
        comorbidity_count = float(np.clip(float(case.get("comorbidity_count", 0.0)) / 5.0, 0.0, 1.0))
        disease_rarity_score = float(np.clip(case.get("disease_rarity_score", 0.0), 0.0, 1.0))
        is_pediatric = 1.0 if age < 18 else 0.0
        is_emergency = 1.0 if urgency == "emergency" else 0.0

        return torch.tensor(
            [
                urgency_level,
                symptom_count,
                red_flag_score,
                patient_age,
                comorbidity_count,
                disease_rarity_score,
                is_pediatric,
                is_emergency,
            ],
            dtype=torch.float32,
        )

    def compute_mcda_score(
        self,
        clinical: torch.Tensor,
        pastwork: torch.Tensor,
        logistics: torch.Tensor,
        trust: torch.Tensor,
    ) -> float:
        """
        Static MCDA score (teacher for distillation).

        Returns: float
        """
        c = clinical.float()
        p = pastwork.float()
        l = logistics.float()
        t = trust.float()

        clinical_score = 0.55 * c[0] + 0.20 * c[1] + 0.15 * c[2] + 0.10 * c[3]
        pastwork_score = 0.30 * p[0] + 0.25 * p[1] + 0.20 * p[2] + 0.15 * p[3] + 0.10 * p[4]
        logistics_score = 0.30 * l[0] + 0.25 * l[1] + 0.20 * l[2] + 0.15 * l[3] + 0.10 * l[4]
        trust_score = 0.50 * t[0] + 0.30 * t[1] + 0.20 * t[2]

        total = 0.40 * clinical_score + 0.25 * pastwork_score + 0.25 * logistics_score + 0.10 * trust_score
        return float(total.item())

    def extract_all_features(
        self,
        doctors_df: pd.DataFrame,
        cases_df: pd.DataFrame,
        doctor_indices: np.ndarray | torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all features for the dataset.

        For each case, extract features for selected doctors.
        """
        if isinstance(doctor_indices, torch.Tensor):
            doctor_indices_np = doctor_indices.detach().cpu().numpy()
        else:
            doctor_indices_np = np.asarray(doctor_indices)

        doctor_embeddings, case_embeddings = self.compute_embeddings(doctors_df, cases_df)

        n_cases = len(cases_df)
        n_candidates = int(doctor_indices_np.shape[1])

        clinical_features = torch.zeros((n_cases, n_candidates, 4), dtype=torch.float32)
        pastwork_features = torch.zeros((n_cases, n_candidates, 5), dtype=torch.float32)
        logistics_features = torch.zeros((n_cases, n_candidates, 5), dtype=torch.float32)
        trust_features = torch.zeros((n_cases, n_candidates, 3), dtype=torch.float32)
        context_features = torch.zeros((n_cases, 8), dtype=torch.float32)
        mcda_scores = torch.zeros((n_cases, n_candidates), dtype=torch.float32)

        for i in range(n_cases):
            case = cases_df.iloc[i]
            case_emb = case_embeddings[i]
            context_features[i] = self.compute_context_features(case)

            for j in range(n_candidates):
                doc_idx = int(doctor_indices_np[i, j])
                doctor = doctors_df.iloc[doc_idx]
                doctor_emb = doctor_embeddings[doc_idx]

                clinical = self.compute_clinical_features(case, doctor, case_emb, doctor_emb)
                pastwork = self.compute_pastwork_features(case, doctor, case_emb, doctor_emb)
                logistics = self.compute_logistics_features(case, doctor)
                trust = self.compute_trust_features(doctor)
                mcda = self.compute_mcda_score(clinical, pastwork, logistics, trust)

                clinical_features[i, j] = clinical
                pastwork_features[i, j] = pastwork
                logistics_features[i, j] = logistics
                trust_features[i, j] = trust
                mcda_scores[i, j] = mcda

        save_dir = Path(".")
        save_dir.mkdir(parents=True, exist_ok=True)

        outputs = {
            "doctor_embeddings": doctor_embeddings,
            "case_embeddings": case_embeddings,
            "clinical_features": clinical_features,
            "pastwork_features": pastwork_features,
            "logistics_features": logistics_features,
            "trust_features": trust_features,
            "context_features": context_features,
            "mcda_scores": mcda_scores,
        }

        torch.save(doctor_embeddings, save_dir / "doctor_embeddings.pt")
        torch.save(case_embeddings, save_dir / "case_embeddings.pt")
        torch.save(clinical_features, save_dir / "clinical_features.pt")
        torch.save(pastwork_features, save_dir / "pastwork_features.pt")
        torch.save(logistics_features, save_dir / "logistics_features.pt")
        torch.save(trust_features, save_dir / "trust_features.pt")
        torch.save(context_features, save_dir / "context_features.pt")
        torch.save(mcda_scores, save_dir / "mcda_scores.pt")

        return outputs
