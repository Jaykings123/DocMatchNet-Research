"""Synthetic healthcare data generation for doctor-patient matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


SPECIALTIES: List[str] = [
    "General Medicine",
    "Cardiology",
    "Neurology",
    "Orthopedics",
    "Dermatology",
    "Pediatrics",
    "Gynecology",
    "Ophthalmology",
    "ENT",
    "Psychiatry",
    "Urology",
    "Nephrology",
    "Pulmonology",
    "Gastroenterology",
    "Endocrinology",
    "Rheumatology",
    "Oncology",
    "Hematology",
    "Infectious Disease",
    "General Surgery",
    "Cardiac Surgery",
    "Neurosurgery",
    "Plastic Surgery",
    "Vascular Surgery",
    "Radiology",
    "Pathology",
    "Anesthesiology",
    "Emergency Medicine",
    "Family Medicine",
    "Sports Medicine",
    "Pain Medicine",
    "Allergy & Immunology",
    "Geriatrics",
    "Neonatology",
    "Hepatology",
    "Critical Care",
    "Palliative Care",
    "Nuclear Medicine",
    "Physical Medicine & Rehabilitation",
    "Preventive Medicine",
    "Sleep Medicine",
    "Tropical Medicine",
]


CITY_COORDS: Dict[str, Tuple[str, float, float]] = {
    "Mumbai": ("Maharashtra", 19.0760, 72.8777),
    "Delhi": ("Delhi", 28.6139, 77.2090),
    "Bangalore": ("Karnataka", 12.9716, 77.5946),
    "Chennai": ("Tamil Nadu", 13.0827, 80.2707),
    "Kolkata": ("West Bengal", 22.5726, 88.3639),
    "Hyderabad": ("Telangana", 17.3850, 78.4867),
    "Pune": ("Maharashtra", 18.5204, 73.8567),
    "Ahmedabad": ("Gujarat", 23.0225, 72.5714),
    "Jaipur": ("Rajasthan", 26.9124, 75.7873),
    "Lucknow": ("Uttar Pradesh", 26.8467, 80.9462),
    "Chandigarh": ("Chandigarh", 30.7333, 76.7794),
    "Kochi": ("Kerala", 9.9312, 76.2673),
    "Bhopal": ("Madhya Pradesh", 23.2599, 77.4126),
    "Patna": ("Bihar", 25.5941, 85.1376),
    "Indore": ("Madhya Pradesh", 22.7196, 75.8577),
    "Nagpur": ("Maharashtra", 21.1458, 79.0882),
    "Coimbatore": ("Tamil Nadu", 11.0168, 76.9558),
    "Vizag": ("Andhra Pradesh", 17.6868, 83.2185),
    "Guwahati": ("Assam", 26.1445, 91.7362),
    "Thiruvananthapuram": ("Kerala", 8.5241, 76.9366),
}

CONTEXTS: Tuple[str, ...] = ("routine", "complex", "rare_disease", "emergency", "pediatric")


@dataclass(frozen=True)
class _CaseProfile:
    target_specialty: str
    symptoms: List[str]
    severity: str
    urgency_level: str
    duration: str
    red_flag_score: float
    comorbidity_count: int
    disease_rarity_score: float


class SyntheticDataGenerator:
    """Generate synthetic doctors, patients, and ground truth matches."""

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config or {}
        self.seed = seed
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.specialties = SPECIALTIES
        self.cities = list(CITY_COORDS.keys())
        self.subspecialty_map = self._build_subspecialty_map()
        self.specialty_symptom_map = self._build_symptom_map()
        self.related_specialties = self._build_related_specialties()
        self.med_topics = self._build_medical_topics()

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.config
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def _build_subspecialty_map(self) -> Dict[str, List[str]]:
        return {
            "General Medicine": ["Internal Medicine", "Primary Care", "Preventive Health"],
            "Cardiology": ["Interventional Cardiology", "Heart Failure", "Electrophysiology"],
            "Neurology": ["Stroke Medicine", "Epilepsy", "Neuroimmunology"],
            "Orthopedics": ["Joint Replacement", "Spine Surgery", "Sports Injury"],
            "Dermatology": ["Dermatosurgery", "Cosmetology", "Pediatric Dermatology"],
            "Pediatrics": ["Pediatric Pulmonology", "Pediatric Neurology", "Adolescent Medicine"],
            "Gynecology": ["Reproductive Endocrinology", "Maternal Fetal Medicine", "Urogynecology"],
            "Ophthalmology": ["Retina", "Cornea", "Glaucoma"],
            "ENT": ["Otology", "Rhinology", "Laryngology"],
            "Psychiatry": ["Addiction Psychiatry", "Child Psychiatry", "Geriatric Psychiatry"],
            "Urology": ["Andrology", "Endourology", "Uro-oncology"],
            "Nephrology": ["Dialysis", "Transplant Nephrology", "Glomerular Disease"],
            "Pulmonology": ["Interventional Pulmonology", "Sleep Respiratory Medicine", "Asthma Care"],
            "Gastroenterology": ["IBD", "Hepatobiliary Disorders", "Therapeutic Endoscopy"],
            "Endocrinology": ["Diabetology", "Thyroid Disorders", "Metabolic Bone Disease"],
            "Rheumatology": ["Autoimmune Disorders", "Spondyloarthropathy", "Vasculitis"],
            "Oncology": ["Medical Oncology", "Solid Tumors", "Precision Oncology"],
            "Hematology": ["Hemostasis", "Leukemia", "Bone Marrow Disorders"],
            "Infectious Disease": ["HIV Medicine", "Hospital Infection", "Travel Medicine"],
            "General Surgery": ["Laparoscopic Surgery", "GI Surgery", "Breast Surgery"],
            "Cardiac Surgery": ["CABG", "Valve Surgery", "Aortic Surgery"],
            "Neurosurgery": ["Spine Neurosurgery", "Tumor Neurosurgery", "Vascular Neurosurgery"],
            "Plastic Surgery": ["Reconstructive Surgery", "Burn Care", "Microvascular Surgery"],
            "Vascular Surgery": ["Peripheral Vascular Disease", "Endovascular Procedures", "Aortic Disease"],
            "Radiology": ["Interventional Radiology", "Neuroradiology", "Musculoskeletal Imaging"],
            "Pathology": ["Histopathology", "Cytopathology", "Molecular Pathology"],
            "Anesthesiology": ["Cardiac Anesthesia", "Neuroanesthesia", "Pain Anesthesia"],
            "Emergency Medicine": ["Trauma Care", "Acute Cardiac Care", "Toxicology"],
            "Family Medicine": ["Preventive Care", "Chronic Disease Care", "Community Health"],
            "Sports Medicine": ["Arthroscopy", "Rehabilitation", "Athlete Performance"],
            "Pain Medicine": ["Interventional Pain", "Neuropathic Pain", "Cancer Pain"],
            "Allergy & Immunology": ["Asthma Allergy", "Food Allergy", "Immunodeficiency"],
            "Geriatrics": ["Cognitive Care", "Frailty Management", "Polypharmacy"],
            "Neonatology": ["NICU Care", "Prematurity", "Neonatal Ventilation"],
            "Hepatology": ["Liver Failure", "Viral Hepatitis", "Cirrhosis Care"],
            "Critical Care": ["ICU Medicine", "Sepsis Management", "Ventilator Care"],
            "Palliative Care": ["End-of-Life Care", "Symptom Relief", "Supportive Oncology"],
            "Nuclear Medicine": ["PET Imaging", "Thyroid Nuclear Therapy", "Radionuclide Studies"],
            "Physical Medicine & Rehabilitation": ["Neurorehabilitation", "Orthorehabilitation", "Spasticity Care"],
            "Preventive Medicine": ["Screening Programs", "Lifestyle Medicine", "Occupational Health"],
            "Sleep Medicine": ["Sleep Apnea", "Insomnia Clinics", "Circadian Disorders"],
            "Tropical Medicine": ["Vector-Borne Disease", "Parasitology", "Travel-related Infection"],
        }

    def _build_symptom_map(self) -> Dict[str, List[str]]:
        return {
            "Cardiology": ["chest pain", "palpitations", "shortness of breath", "ankle swelling", "fatigue"],
            "Neurology": ["headache", "dizziness", "seizure", "weakness", "numbness"],
            "Orthopedics": ["knee pain", "back pain", "joint stiffness", "shoulder pain", "sports injury"],
            "Dermatology": ["itchy rash", "acne flare", "skin lesion", "eczema", "hair loss"],
            "Pediatrics": ["fever in child", "poor feeding", "persistent cough", "vomiting", "skin rash in child"],
            "Gynecology": ["irregular periods", "pelvic pain", "abnormal discharge", "heavy bleeding", "infertility concerns"],
            "Ophthalmology": ["blurred vision", "eye redness", "eye pain", "dry eyes", "floaters"],
            "ENT": ["sore throat", "sinus congestion", "hearing loss", "ear pain", "hoarseness"],
            "Psychiatry": ["anxiety", "low mood", "sleep disturbance", "panic episodes", "poor concentration"],
            "Urology": ["burning urination", "frequent urination", "flank pain", "blood in urine", "urinary urgency"],
            "Nephrology": ["leg swelling", "decreased urine output", "foamy urine", "high creatinine", "fatigue"],
            "Pulmonology": ["chronic cough", "wheezing", "breathlessness", "chest tightness", "snoring"],
            "Gastroenterology": ["abdominal pain", "bloating", "acid reflux", "constipation", "diarrhea"],
            "Endocrinology": ["weight gain", "weight loss", "increased thirst", "heat intolerance", "fatigue"],
            "Rheumatology": ["joint pain", "morning stiffness", "joint swelling", "fatigue", "muscle pain"],
            "Oncology": ["unexplained weight loss", "persistent lump", "fatigue", "night sweats", "loss of appetite"],
            "Infectious Disease": ["prolonged fever", "body ache", "night sweats", "persistent cough", "diarrhea"],
            "Emergency Medicine": ["severe chest pain", "breathing difficulty", "loss of consciousness", "major trauma", "severe bleeding"],
            "Family Medicine": ["fever", "cold and cough", "body ache", "mild headache", "general weakness"],
            "Geriatrics": ["memory decline", "recurrent falls", "poor appetite", "polypharmacy issue", "frailty"],
            "Neonatology": ["jaundice in newborn", "breathing distress in newborn", "poor suckling", "low birth weight concern"],
            "Hepatology": ["jaundice", "abdominal swelling", "right upper abdominal pain", "loss of appetite"],
            "Critical Care": ["septic shock signs", "respiratory failure", "multi-organ dysfunction", "altered sensorium"],
            "Sleep Medicine": ["loud snoring", "daytime sleepiness", "insomnia", "frequent night awakenings"],
            "Tropical Medicine": ["high-grade fever", "travel-associated rash", "persistent diarrhea", "mosquito exposure illness"],
        }

    def _build_related_specialties(self) -> Dict[str, List[str]]:
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
        related: Dict[str, List[str]] = {spec: [] for spec in self.specialties}
        for group in groups:
            for spec in group:
                related[spec] = [x for x in group if x != spec]
        return related

    def _build_medical_topics(self) -> List[str]:
        return [
            "chronic disease management",
            "telemedicine workflows",
            "patient safety",
            "clinical decision support",
            "population health analytics",
            "precision medicine",
            "evidence-based guidelines",
            "care coordination",
            "preventive screening",
            "quality improvement",
            "health equity",
            "digital therapeutics",
        ]

    def _sample_city_location(self) -> Dict[str, Any]:
        city = self.rng.choice(self.cities)
        state, lat, lon = CITY_COORDS[city]
        jitter_lat = lat + float(self.rng.normal(0.0, 0.05))
        jitter_lon = lon + float(self.rng.normal(0.0, 0.05))
        return {"city": city, "state": state, "lat": round(jitter_lat, 4), "lon": round(jitter_lon, 4)}

    def _sample_languages(self, max_count: int = 4) -> List[str]:
        all_languages = [
            "English",
            "Hindi",
            "Tamil",
            "Telugu",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Gujarati",
            "Bengali",
            "Punjabi",
        ]
        n_lang = int(self.rng.integers(1, max_count + 1))
        return list(self.rng.choice(all_languages, size=n_lang, replace=False))

    def _doctor_name(self) -> str:
        first_names = [
            "Aarav",
            "Vivaan",
            "Aditya",
            "Arjun",
            "Ishaan",
            "Ananya",
            "Diya",
            "Kiara",
            "Meera",
            "Riya",
            "Saanvi",
            "Nisha",
            "Rahul",
            "Priya",
            "Karan",
            "Neha",
            "Vikram",
            "Sneha",
        ]
        last_names = [
            "Sharma",
            "Patel",
            "Reddy",
            "Iyer",
            "Menon",
            "Singh",
            "Gupta",
            "Nair",
            "Kulkarni",
            "Chatterjee",
            "Bose",
            "Mehta",
            "Agarwal",
            "Mishra",
            "Jain",
            "Kapoor",
        ]
        return f"Dr. {self.rng.choice(first_names)} {self.rng.choice(last_names)}"

    def generate_doctors(self, n_doctors: int = 500) -> pd.DataFrame:
        """
        Generate synthetic doctor profiles.

        Returns: DataFrame with all doctors.
        """
        qualifications_pool = ["MBBS", "MD", "MS", "DM", "MCh", "DNB", "FRCP", "Fellowship"]
        availability_modes = ["online", "in-person", "both"]
        hospitals = [
            "Apollo Hospitals",
            "Fortis Healthcare",
            "Manipal Hospitals",
            "Narayana Health",
            "Aster Medcity",
            "Max Healthcare",
            "AIIMS Network",
            "KIMS Hospitals",
            "Medanta",
            "CARE Hospitals",
        ]

        rows: List[Dict[str, Any]] = []
        for doctor_id in range(n_doctors):
            doctor_name = self._doctor_name()
            specialty = str(self.rng.choice(self.specialties))
            all_subspecialties = self.subspecialty_map.get(specialty, [])
            n_subspec = int(self.rng.integers(0, min(3, len(all_subspecialties)) + 1))
            subspecialties = list(self.rng.choice(all_subspecialties, size=n_subspec, replace=False)) if n_subspec else []

            years_experience = int(self.rng.integers(1, 41))
            n_quals = int(self.rng.integers(1, 4))
            qualifications = list(self.rng.choice(qualifications_pool, size=n_quals, replace=False))
            location = self._sample_city_location()

            fee_base = float(self.rng.uniform(200, 5000))
            if years_experience > 20:
                fee_base *= 1.15
            if specialty in {"Cardiology", "Neurology", "Oncology", "Neurosurgery", "Cardiac Surgery"}:
                fee_base *= 1.2
            consultation_fee = int(np.clip(fee_base, 200, 5000))

            available_modes = str(self.rng.choice(availability_modes, p=[0.2, 0.3, 0.5]))
            availability_score = float(np.clip(self.rng.normal(0.72, 0.15), 0.0, 1.0))
            nmc_verified = bool(self.rng.random() < 0.9)
            profile_completeness = float(np.clip(self.rng.normal(0.78, 0.18), 0.3, 1.0))
            review_score = float(np.clip(self.rng.normal(4.2, 0.55), 2.5, 5.0))
            num_reviews = int(np.clip(self.rng.negative_binomial(5, 0.05), 0, 500))
            publications_count = int(np.clip(self.rng.poisson(6 if years_experience < 8 else 12), 0, 100))
            n_topics = int(self.rng.integers(1, 4))
            research_topics = list(self.rng.choice(self.med_topics, size=n_topics, replace=False))
            n_hosp = int(self.rng.integers(1, 3))
            hospital_affiliations = list(self.rng.choice(hospitals, size=n_hosp, replace=False))
            consultation_completion_rate = float(np.clip(self.rng.normal(0.9, 0.08), 0.7, 1.0))
            languages = self._sample_languages()

            desc = (
                f"{doctor_name} is a {specialty} specialist with {years_experience} years of experience. "
                f"Focus areas include {', '.join(subspecialties) if subspecialties else 'comprehensive clinical care'}. "
                f"Affiliated with {hospital_affiliations[0]} and known for a patient-centered approach, "
                f"high follow-up completion ({consultation_completion_rate:.2f}), and evidence-based treatment."
            )

            rows.append(
                {
                    "doctor_id": f"D{doctor_id:04d}",
                    "name": doctor_name,
                    "specialty": specialty,
                    "subspecialties": subspecialties,
                    "years_experience": years_experience,
                    "qualifications": qualifications,
                    "languages": languages,
                    "location": location,
                    "consultation_fee": consultation_fee,
                    "available_modes": available_modes,
                    "availability_score": availability_score,
                    "nmc_verified": nmc_verified,
                    "profile_completeness": profile_completeness,
                    "review_score": review_score,
                    "num_reviews": num_reviews,
                    "publications_count": publications_count,
                    "research_topics": research_topics,
                    "hospital_affiliations": hospital_affiliations,
                    "consultation_completion_rate": consultation_completion_rate,
                    "expertise_description": desc,
                }
            )
        return pd.DataFrame(rows)

    def _sample_context(self, n_cases: int) -> np.ndarray:
        p = self._cfg("contexts", default={}) or {}
        weights = np.array(
            [
                float(p.get("routine", 0.60)),
                float(p.get("complex", 0.15)),
                float(p.get("rare_disease", 0.10)),
                float(p.get("emergency", 0.10)),
                float(p.get("pediatric", 0.05)),
            ],
            dtype=float,
        )
        weights = weights / weights.sum()
        return self.rng.choice(CONTEXTS, size=n_cases, p=weights)

    def _sample_case_profile(self, context_category: str) -> _CaseProfile:
        if context_category == "pediatric":
            target_specialty = "Pediatrics"
            severity = str(self.rng.choice(["mild", "moderate", "severe"], p=[0.5, 0.35, 0.15]))
            urgency = str(self.rng.choice(["routine", "semi-urgent", "urgent"], p=[0.45, 0.35, 0.20]))
            comorbidity = int(self.rng.integers(0, 2))
            rarity = float(np.clip(self.rng.normal(0.3, 0.15), 0.0, 1.0))
        elif context_category == "emergency":
            target_specialty = str(self.rng.choice(["Emergency Medicine", "Critical Care", "Cardiology", "Neurology"]))
            severity = str(self.rng.choice(["severe", "critical"], p=[0.55, 0.45]))
            urgency = "emergency"
            comorbidity = int(self.rng.integers(1, 4))
            rarity = float(np.clip(self.rng.normal(0.4, 0.2), 0.0, 1.0))
        elif context_category == "rare_disease":
            target_specialty = str(
                self.rng.choice(["Hematology", "Oncology", "Neurology", "Rheumatology", "Infectious Disease"])
            )
            severity = str(self.rng.choice(["moderate", "severe"], p=[0.45, 0.55]))
            urgency = str(self.rng.choice(["semi-urgent", "urgent"], p=[0.5, 0.5]))
            comorbidity = int(self.rng.integers(1, 5))
            rarity = float(np.clip(self.rng.normal(0.82, 0.12), 0.0, 1.0))
        elif context_category == "complex":
            target_specialty = str(
                self.rng.choice(
                    ["Cardiology", "Neurology", "Nephrology", "Endocrinology", "Pulmonology", "Gastroenterology"]
                )
            )
            severity = str(self.rng.choice(["moderate", "severe"], p=[0.6, 0.4]))
            urgency = str(self.rng.choice(["routine", "semi-urgent", "urgent"], p=[0.2, 0.5, 0.3]))
            comorbidity = int(self.rng.integers(2, 6))
            rarity = float(np.clip(self.rng.normal(0.45, 0.2), 0.0, 1.0))
        else:
            target_specialty = str(
                self.rng.choice(
                    [
                        "General Medicine",
                        "Family Medicine",
                        "Dermatology",
                        "Orthopedics",
                        "ENT",
                        "Gynecology",
                        "Ophthalmology",
                    ]
                )
            )
            severity = str(self.rng.choice(["mild", "moderate", "severe"], p=[0.6, 0.3, 0.1]))
            urgency = str(self.rng.choice(["routine", "semi-urgent"], p=[0.75, 0.25]))
            comorbidity = int(self.rng.integers(0, 3))
            rarity = float(np.clip(self.rng.normal(0.2, 0.12), 0.0, 1.0))

        duration_units = ["days", "weeks", "months"]
        unit = str(self.rng.choice(duration_units, p=[0.55, 0.35, 0.10]))
        max_n = {"days": 20, "weeks": 12, "months": 6}[unit]
        duration = f"{int(self.rng.integers(1, max_n + 1))} {unit}"
        red_flag = float(np.clip(self.rng.normal(0.25, 0.15), 0.0, 1.0))
        if severity in {"severe", "critical"}:
            red_flag = float(np.clip(red_flag + 0.35, 0.0, 1.0))
        if context_category == "emergency":
            red_flag = float(np.clip(red_flag + 0.25, 0.0, 1.0))

        symptoms_pool = self.specialty_symptom_map.get(target_specialty, self.specialty_symptom_map["Family Medicine"])
        n_symptoms = int(self.rng.integers(1, 9))
        symptoms = list(self.rng.choice(symptoms_pool, size=min(n_symptoms, len(symptoms_pool)), replace=False))
        if len(symptoms) < n_symptoms:
            filler = list(self.rng.choice(self.specialty_symptom_map["Family Medicine"], size=n_symptoms - len(symptoms)))
            symptoms.extend(filler)

        return _CaseProfile(
            target_specialty=target_specialty,
            symptoms=symptoms,
            severity=severity,
            urgency_level=urgency,
            duration=duration,
            red_flag_score=red_flag,
            comorbidity_count=comorbidity,
            disease_rarity_score=rarity,
        )

    def _symptom_description(self, specialty: str, symptoms: Sequence[str], severity: str, duration: str) -> str:
        tone_map = {
            "mild": "The symptoms are present but manageable",
            "moderate": "Symptoms are persistent and affecting daily activities",
            "severe": "Symptoms are significantly impairing routine functioning",
            "critical": "Symptoms are acute and potentially life-threatening",
        }
        opening = tone_map.get(severity, "Symptoms are clinically relevant")
        lead = ", ".join(symptoms[:3])
        additional = ""
        if len(symptoms) > 3:
            additional = f" with additional concerns including {', '.join(symptoms[3:5])}"
        return (
            f"Patient reports {lead}{additional} for {duration}. "
            f"{opening}. Preliminary triage indicates likely need for {specialty} evaluation."
        )

    def generate_patient_cases(self, n_cases: int = 15000) -> pd.DataFrame:
        """
        Generate synthetic patient cases with realistic context distribution.

        Returns: DataFrame with all cases.
        """
        genders = ["M", "F", "Other"]
        preferred_modes = ["online", "in-person", "no-preference"]
        allergies_pool = ["penicillin", "dust", "pollen", "peanut", "none", "seafood", "latex"]
        history_pool = [
            "hypertension",
            "type 2 diabetes",
            "asthma",
            "hypothyroidism",
            "chronic kidney disease",
            "ischemic heart disease",
            "migraine",
            "arthritis",
            "none",
        ]
        medication_pool = [
            "metformin",
            "amlodipine",
            "atorvastatin",
            "levothyroxine",
            "pantoprazole",
            "paracetamol",
            "inhaled budesonide",
            "none",
        ]

        contexts = self._sample_context(n_cases)
        rows: List[Dict[str, Any]] = []
        for case_id in range(n_cases):
            context = str(contexts[case_id])
            profile = self._sample_case_profile(context)
            location = self._sample_city_location()
            age = int(self.rng.integers(0, 91)) if context != "pediatric" else int(self.rng.integers(0, 18))
            preferred_mode = str(self.rng.choice(preferred_modes, p=[0.35, 0.30, 0.35]))
            preferred_language = str(self.rng.choice(self._sample_languages(max_count=6)))
            n_history = int(self.rng.integers(0, 4))
            n_meds = int(self.rng.integers(0, 4))
            n_allergies = int(self.rng.integers(0, 3))
            medical_history = list(self.rng.choice(history_pool, size=n_history, replace=False))
            current_medications = list(self.rng.choice(medication_pool, size=n_meds, replace=False))
            allergies = list(self.rng.choice(allergies_pool, size=n_allergies, replace=False))
            med_hist = [x for x in medical_history if x != "none"]
            meds = [x for x in current_medications if x != "none"]
            allg = [x for x in allergies if x != "none"]

            budget_min = int(self.rng.integers(200, 2500))
            budget_max = int(min(6000, budget_min + int(self.rng.integers(500, 3500))))

            rows.append(
                {
                    "case_id": f"C{case_id:05d}",
                    "patient_age": age,
                    "patient_gender": str(self.rng.choice(genders, p=[0.49, 0.49, 0.02])),
                    "symptoms": profile.symptoms,
                    "symptom_description": self._symptom_description(
                        profile.target_specialty, profile.symptoms, profile.severity, profile.duration
                    ),
                    "duration": profile.duration,
                    "severity": profile.severity,
                    "medical_history": med_hist,
                    "current_medications": meds,
                    "allergies": allg,
                    "location": location,
                    "preferred_language": preferred_language,
                    "preferred_mode": preferred_mode,
                    "budget_range": [budget_min, budget_max],
                    "urgency_level": profile.urgency_level,
                    "context_category": context,
                    "target_specialty": profile.target_specialty,
                    "red_flag_score": profile.red_flag_score,
                    "comorbidity_count": profile.comorbidity_count,
                    "disease_rarity_score": profile.disease_rarity_score,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        r = 6371.0
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return r * c

    def generate_relevance_labels(
        self,
        doctors_df: pd.DataFrame,
        cases_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ground truth relevance labels (0-4 scale) for each doctor-case pair.

        Returns:
        - relevance_matrix: (n_cases, 100)
        - doctor_indices: (n_cases, 100)
        """
        n_cases = len(cases_df)
        n_doctors = len(doctors_df)
        candidate_k = min(100, n_doctors)
        top_k = min(50, n_doctors)
        random_k = max(0, candidate_k - top_k)

        specialties = doctors_df["specialty"].to_numpy()
        availability = doctors_df["availability_score"].to_numpy(dtype=float)
        completion = doctors_df["consultation_completion_rate"].to_numpy(dtype=float)
        review = doctors_df["review_score"].to_numpy(dtype=float)
        nmc = doctors_df["nmc_verified"].to_numpy(dtype=bool)
        fees = doctors_df["consultation_fee"].to_numpy(dtype=float)
        modes = doctors_df["available_modes"].to_numpy()
        doc_lat = doctors_df["location"].apply(lambda x: x["lat"]).to_numpy(dtype=float)
        doc_lon = doctors_df["location"].apply(lambda x: x["lon"]).to_numpy(dtype=float)

        relevance_matrix = np.zeros((n_cases, candidate_k), dtype=np.int64)
        doctor_indices = np.zeros((n_cases, candidate_k), dtype=np.int64)

        fee_norm = (fees - fees.min()) / (fees.max() - fees.min() + 1e-8)
        review_norm = (review - 2.5) / (5.0 - 2.5)

        for i, case in cases_df.iterrows():
            target_specialty = case["target_specialty"]
            pref_mode = case["preferred_mode"]
            budget_min, budget_max = case["budget_range"]

            exact = specialties == target_specialty
            related_specs = self.related_specialties.get(target_specialty, [])
            related = np.isin(specialties, related_specs)
            broad_related = np.isin(specialties, self.related_specialties.get(target_specialty, []) + [target_specialty])

            lat1, lon1 = case["location"]["lat"], case["location"]["lon"]
            distances = self._haversine_km(float(lat1), float(lon1), doc_lat, doc_lon)
            nearby = distances < 35.0
            mode_ok = (pref_mode == "no-preference") | (modes == "both") | (modes == pref_mode)
            availability_ok = availability > 0.6
            mostly_available = availability > 0.45
            budget_ok = (fees >= float(budget_min)) & (fees <= float(budget_max))
            high_quality = (completion > 0.85) & (review > 4.0) & nmc

            score = (
                3.0 * exact.astype(float)
                + 1.5 * related.astype(float)
                + 1.0 * availability
                + 0.7 * completion
                + 0.5 * review_norm
                + 0.3 * nmc.astype(float)
                + 0.25 * budget_ok.astype(float)
                + 0.25 * mode_ok.astype(float)
                + 0.25 * high_quality.astype(float)
                - 0.8 * np.clip(distances / 600.0, 0.0, 1.5)
                - 0.2 * fee_norm
            )

            top_candidates = np.argpartition(-score, top_k - 1)[:top_k] if top_k > 0 else np.array([], dtype=int)
            if random_k > 0:
                remaining = np.setdiff1d(np.arange(n_doctors), top_candidates, assume_unique=False)
                sampled = (
                    self.rng.choice(remaining, size=random_k, replace=False) if len(remaining) >= random_k else remaining
                )
                selected = np.concatenate([top_candidates, sampled])
            else:
                selected = top_candidates

            if len(selected) < candidate_k:
                pad = np.setdiff1d(np.arange(n_doctors), selected, assume_unique=False)[: (candidate_k - len(selected))]
                selected = np.concatenate([selected, pad])

            self.rng.shuffle(selected)
            selected = selected[:candidate_k]
            doctor_indices[i] = selected

            sel_exact = exact[selected]
            sel_related = related[selected]
            sel_broad = broad_related[selected]
            sel_mode_ok = mode_ok[selected]
            sel_nearby = nearby[selected]
            sel_avail = availability[selected]

            rel = np.zeros(candidate_k, dtype=np.int64)
            rel[(~sel_exact) & (~sel_broad)] = 0
            rel[(~sel_exact) & sel_broad] = 1
            rel[(sel_exact & (~sel_mode_ok)) | (sel_exact & (sel_avail <= 0.45)) | (sel_related & (~sel_nearby))] = 2
            rel[sel_exact & sel_mode_ok & (sel_avail > 0.45)] = 3
            rel[sel_exact & sel_mode_ok & (sel_avail > 0.6) & sel_nearby] = 4
            relevance_matrix[i] = rel

        return relevance_matrix, doctor_indices

    def generate_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Generate complete dataset."""
        n_doctors = int(self._cfg("data", "n_doctors", default=500))
        n_cases = int(self._cfg("data", "n_cases", default=15000))
        doctors = self.generate_doctors(n_doctors=n_doctors)
        cases = self.generate_patient_cases(n_cases=n_cases)
        relevance, indices = self.generate_relevance_labels(doctors, cases)
        return doctors, cases, relevance, indices
