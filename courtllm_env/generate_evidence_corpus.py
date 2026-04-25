# generate_evidence_corpus.py — Generate 5000 synthetic evidence entries

import json
import random
from pathlib import Path

DOMAINS = ["biomedical", "historical", "legal", "financial", "scientific"]

# Domain-specific templates
TEMPLATES = {
    "biomedical": [
        "Studies show that {treatment} reduces {condition} risk by approximately {percent}%",
        "The {organ} contains approximately {number} {unit}",
        "{drug} was discovered by {person} in {year}",
        "Research indicates {protein} plays a key role in {process}",
        "{disease} affects approximately {number} people worldwide annually",
        "Clinical trials demonstrate {therapy} improves {outcome} by {percent}%",
        "The {gene} gene is associated with increased risk of {condition}",
        "{vitamin} deficiency can lead to {symptom}",
        "Patients with {condition} typically experience {symptom}",
        "{procedure} has a success rate of approximately {percent}%"
    ],
    "historical": [
        "{event} occurred in {year}",
        "{person} was born in {year} in {location}",
        "The {treaty} was signed in {year}",
        "{war} lasted from {year} to {year2}",
        "{leader} ruled {country} from {year} to {year2}",
        "The {movement} began in {year}",
        "{invention} was invented by {person} in {year}",
        "{battle} took place in {year} near {location}",
        "The {empire} fell in {year}",
        "{document} was written in {year}"
    ],
    "legal": [
        "The {law} was enacted in {year}",
        "{case} established the precedent for {principle}",
        "The {amendment} was ratified in {year}",
        "{court} has {number} justices",
        "{statute} requires {requirement}",
        "Under {law}, {action} is classified as {classification}",
        "The statute of limitations for {crime} is {number} years",
        "{right} was established in {year}",
        "{doctrine} states that {principle}",
        "The {act} prohibits {action}"
    ],
    "financial": [
        "The {index} tracks {number} companies",
        "{event} occurred in {year}",
        "The {institution} was established in {year}",
        "{currency} was introduced in {year}",
        "The {crisis} was triggered by {cause}",
        "{regulation} requires {requirement}",
        "The {rate} is set by {institution}",
        "{market} capitalization reached ${number} billion in {year}",
        "{company} was founded in {year}",
        "The {ratio} measures {metric}"
    ],
    "scientific": [
        "The speed of {particle} is approximately {number} {unit}",
        "{element} has an atomic number of {number}",
        "{theory} was proposed by {person} in {year}",
        "{constant} equals approximately {number}",
        "{phenomenon} occurs when {condition}",
        "The {law} states that {principle}",
        "{compound} has the chemical formula {formula}",
        "{process} converts {input} into {output}",
        "The {structure} was discovered in {year}",
        "{measurement} is defined as {definition}"
    ]
}

# Vocabulary for filling templates
VOCAB = {
    "treatment": ["aspirin", "chemotherapy", "radiation therapy", "immunotherapy", "surgery"],
    "condition": ["heart disease", "cancer", "diabetes", "stroke", "infection"],
    "organ": ["human brain", "liver", "heart", "kidney", "lung"],
    "drug": ["Penicillin", "Insulin", "Aspirin", "Morphine", "Quinine"],
    "person": ["Alexander Fleming", "Marie Curie", "Louis Pasteur", "Jonas Salk", "Albert Einstein"],
    "disease": ["Malaria", "Tuberculosis", "Influenza", "COVID-19", "Measles"],
    "protein": ["hemoglobin", "insulin", "collagen", "antibodies", "enzymes"],
    "process": ["cell division", "metabolism", "immune response", "DNA replication", "protein synthesis"],
    "gene": ["BRCA1", "TP53", "APOE", "CFTR", "HTT"],
    "therapy": ["cognitive behavioral therapy", "physical therapy", "gene therapy", "hormone therapy"],
    "outcome": ["survival rates", "quality of life", "symptom severity", "recovery time"],
    "vitamin": ["Vitamin D", "Vitamin C", "Vitamin B12", "Vitamin A", "Folic acid"],
    "symptom": ["fatigue", "pain", "inflammation", "cognitive decline", "weakness"],
    "procedure": ["coronary bypass surgery", "hip replacement", "cataract surgery", "appendectomy"],

    "event": ["World War II", "The French Revolution", "The Industrial Revolution", "The Renaissance"],
    "location": ["London", "Paris", "Rome", "Athens", "Beijing"],
    "treaty": ["Treaty of Versailles", "Treaty of Paris", "Treaty of Westphalia", "Treaty of Rome"],
    "war": ["World War I", "The Napoleonic Wars", "The Hundred Years War", "The Civil War"],
    "leader": ["Napoleon Bonaparte", "Julius Caesar", "Queen Victoria", "Genghis Khan"],
    "country": ["France", "England", "Rome", "China", "Egypt"],
    "movement": ["Civil Rights Movement", "Women's Suffrage Movement", "Enlightenment", "Reformation"],
    "invention": ["The printing press", "The steam engine", "The telephone", "The light bulb"],
    "battle": ["Battle of Waterloo", "Battle of Gettysburg", "Battle of Hastings", "Battle of Stalingrad"],
    "empire": ["Roman Empire", "Ottoman Empire", "British Empire", "Mongol Empire"],
    "document": ["The Magna Carta", "The Constitution", "The Declaration of Independence"],

    "law": ["Civil Rights Act", "Sherman Antitrust Act", "Clean Air Act", "Americans with Disabilities Act"],
    "case": ["Brown v. Board of Education", "Roe v. Wade", "Miranda v. Arizona", "Marbury v. Madison"],
    "principle": ["equal protection", "due process", "freedom of speech", "right to privacy"],
    "amendment": ["First Amendment", "Fourteenth Amendment", "Nineteenth Amendment"],
    "court": ["Supreme Court", "Court of Appeals", "District Court"],
    "statute": ["Title VII", "Section 1983", "Fair Labor Standards Act"],
    "action": ["discrimination", "harassment", "retaliation", "breach of contract"],
    "classification": ["a felony", "a misdemeanor", "a civil violation", "an infraction"],
    "crime": ["murder", "fraud", "theft", "assault"],
    "right": ["Miranda rights", "Voting rights", "Property rights", "Privacy rights"],
    "doctrine": ["Stare decisis", "Res judicata", "Qualified immunity"],
    "act": ["Securities Act", "Sarbanes-Oxley Act", "Dodd-Frank Act"],

    "index": ["Dow Jones Industrial Average", "S&P 500", "NASDAQ Composite", "FTSE 100"],
    "institution": ["Federal Reserve", "World Bank", "International Monetary Fund", "Bank of England"],
    "currency": ["Euro", "Dollar", "Pound Sterling", "Yen"],
    "crisis": ["2008 financial crisis", "Great Depression", "Dot-com bubble", "Asian financial crisis"],
    "cause": ["subprime mortgages", "speculation", "bank failures", "currency devaluation"],
    "regulation": ["Basel III", "Dodd-Frank", "MiFID II", "Sarbanes-Oxley"],
    "requirement": ["capital adequacy ratios", "stress testing", "disclosure requirements"],
    "rate": ["federal funds rate", "discount rate", "prime rate", "LIBOR"],
    "market": ["US stock market", "bond market", "cryptocurrency market", "commodities market"],
    "company": ["Apple", "Microsoft", "Amazon", "Google", "Tesla"],
    "ratio": ["price-to-earnings ratio", "debt-to-equity ratio", "current ratio"],
    "metric": ["profitability", "liquidity", "leverage", "efficiency"],

    "particle": ["light", "sound", "electrons", "neutrinos"],
    "element": ["Carbon", "Oxygen", "Hydrogen", "Iron", "Gold"],
    "theory": ["Theory of Relativity", "Quantum Theory", "Evolution", "Big Bang Theory"],
    "constant": ["Planck's constant", "Gravitational constant", "Boltzmann constant"],
    "phenomenon": ["photosynthesis", "nuclear fusion", "superconductivity", "quantum entanglement"],
    "condition": ["temperature drops below critical point", "pressure exceeds threshold", "energy is absorbed"],
    "compound": ["Water", "Carbon dioxide", "Methane", "Glucose", "DNA"],
    "formula": ["H2O", "CO2", "CH4", "C6H12O6", "NaCl"],
    "input": ["glucose", "carbon dioxide", "nitrogen", "sunlight"],
    "output": ["ATP", "oxygen", "ammonia", "chemical energy"],
    "structure": ["DNA double helix", "atomic structure", "cell membrane", "protein folding"],
    "measurement": ["meter", "kilogram", "second", "ampere", "kelvin"],
    "definition": ["the distance light travels in vacuum", "the mass of the prototype", "the duration of radiation periods"],

    "unit": ["neurons", "cells", "genes", "base pairs", "molecules"],
    "percent": [str(random.randint(10, 95)) for _ in range(20)],
    "number": [str(random.randint(1, 999)) for _ in range(50)],
    "year": [str(random.randint(1800, 2024)) for _ in range(100)],
    "year2": [str(random.randint(1800, 2024)) for _ in range(100)]
}

def generate_entry(domain: str, idx: int) -> dict:
    """Generate a single evidence entry"""
    template = random.choice(TEMPLATES[domain])

    # Fill template with vocabulary
    snippet = template
    for key in VOCAB:
        if f"{{{key}}}" in snippet:
            snippet = snippet.replace(f"{{{key}}}", random.choice(VOCAB[key]), 1)

    return {
        "source_id": f"{domain.upper()}_{idx:04d}",
        "title": f"{domain.title()} Source {idx}",
        "domain": domain,
        "snippet": snippet,
        "url": f"https://example.com/{domain}/{idx}",
        "is_verified": True
    }

def generate_corpus(total_entries: int = 5000) -> list:
    """Generate full evidence corpus"""
    corpus = []
    entries_per_domain = total_entries // len(DOMAINS)

    for domain in DOMAINS:
        for i in range(entries_per_domain):
            corpus.append(generate_entry(domain, i))

    # Fill remaining entries
    while len(corpus) < total_entries:
        domain = random.choice(DOMAINS)
        corpus.append(generate_entry(domain, len(corpus)))

    return corpus

if __name__ == "__main__":
    print("Generating 5000 evidence corpus entries...")
    corpus = generate_corpus(5000)

    output_path = Path(__file__).parent / "data" / "evidence_corpus.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in corpus:
            f.write(json.dumps(entry) + '\n')

    print(f"✓ Generated {len(corpus)} entries")
    print(f"✓ Saved to {output_path}")
