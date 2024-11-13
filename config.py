RELEVANT_CATEGORIES = {"AGE", "SEX", "SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER"}
DIM = 1024
PREDEFINED_SEX_MAPPING = {
    "female": True,
    "woman": True,
    "girl": True,
    "lady": True,
    "mother": True,
    "bride": True,

    "male": False,
    "man": False,
    "boy": False,
    "gentleman": False,
    "males": False,
    "groom": False,

    "neonate": None,
    "infant": None,
    "baby": None,
    "newborn": None,
    "child": None,
    "teenager": None,
    "youth": None
}

TEST_QUERIES = [
    {
        "question": "A person of eonism derives pleasure from.",
        "answer": "Wearing clothes of opposite sex",
        "answer_index": "opa",
        "opa": "Wearing clothes of opposite sex",
        "opb": "Fondling female body pas",
        "opc": "Rubbing genitalia against body of other person",
        "opd": "Seeing the opposite paner nude",
        "generated_explanation": "A person of eonism derives pleasure from. Wearing clothes of opposite sex. Here is "
                                 "an additional explanation: Ans. a. Wearing clothes of opposite sexIt is a person "
                                 "whose personality is dominated by the desire to be identified with the opposite sex "
                                 "It is usuallyfound in males who obtain sexual pleasure by wearingfemale dress."
    },
    {
        "question": "18 year old female presents with an ovarian mass, her serum bio marker are found to be normal "
                    "except for LDH, which is found to be elevated. The most likely diagnosis is:",
        "answer": "Dysgerminoma",
        "answer_index": "opa",
        "opa": "Dysgerminoma",
        "opb": "Endodermal sinus tumor",
        "opc": "Malignant terratoma",
        "opd": "Mucinous cystadeno carcinoma",
        "generated_explanation": "18 year old female presents with an ovarian mass, her serum bio marker are found to "
                                 "be normal except for LDH, which is found to be elevated. The most likely diagnosis "
                                 "is: Dysgerminoma. Here is an additional explanation: NOTE :- * Young girls with - "
                                 "Germ cell tumor Ovarian mass * Old women with - Epithelial serous tumor Ovarian "
                                 "mass Biomarkers Dysgerminoma | LDH, | placental alkaline Po4 Endodermal sinus tumor "
                                 "a feto protein and antitrypsin"
    },
    {
        "question": "Klenow fragment is formed by loss of fragment having which activity:",
        "answer": "5'- 3' exonuclease",
        "answer_index": "opc",
        "opa": "5'- 3' polymerase",
        "opb": "3'- 5' exonuclease",
        "opc": "5'- 3' exonuclease",
        "opd": "3'- 5' polymerase",
        "generated_explanation": "Klenow fragment is formed by loss of fragment having which activity: 5'- 3' "
                                 "exonuclease. Here is an additional explanation: Klenow fragment Large fragment "
                                 "produced by Subtilisin mediated proteolytic cleavage of E.Coli DNA polymerase I. "
                                 "Proteolysis removes the 5' -->3' exonuclease activity from N-terminal. Klenow "
                                 "fragment - Functions : Remove 3' overhang Fills 5' overhangs Synthesis of "
                                 "double-stranded DNA from single-stranded templates Preparation of radioactive DNA "
                                 "probes Was used in PCR"
    },
    {
        "question": "Vaccine for caries is based on which immunoglobulin",
        "answer": "IgA",
        "answer_index": "opb",
        "opa": "IgG",
        "opb": "IgA",
        "opc": "IgE",
        "opd": "IgM",
        "generated_explanation": "Vaccine for caries is based on which immunoglobulin IgA. "
    },
    {
        "question": "Which of the following muscle is not supplied by the nerve marked in the diagram?",
        "answer": "Superior oblique",
        "answer_index": "opa",
        "opa": "Superior oblique",
        "opb": "Medial rectus",
        "opc": "Inferior rectus",
        "opd": "Inferior oblique",
        "generated_explanation": "Which of the following muscle is not supplied by the nerve marked in the diagram? "
                                 "Superior oblique. Here is an additional explanation: The nerve marked in the "
                                 "diagram is oculomotor nerve. It supplies superior rectus, inferior rectus, "
                                 "medial rectus and inferior oblique. Superior oblique is supplied by Trochlear nerve."
    }, {
        "question": "A 45-year-old man comes to the physician for a routine health maintenance examination. He was "
                    "diagnosed with HIV 15 years ago. He was taking triple antiretroviral therapy but stopped a few "
                    "months ago because he was feeling well. He lives in Wyoming. Vital signs are within normal limits. "
                    "Cardiopulmonary examination shows no abnormalities. His CD4+ T-lymphocyte count is 47/mm3 (N \u2265 "
                    "500). The patient currently refuses to restart antiretroviral therapy. Which of the following "
                    "medication regimens is most appropriate at this time?",
        "answer": "Trimethoprim, sulfamethoxazole, azithromycin",
        "answer_idx": "opc",
        "opa": "Azithromycin and itraconazole",
        "opb": "Azithromycin and amphotericin B",
        "opc": "Trimethoprim, sulfamethoxazole, azithromycin",
        "opd": "Dapsone, pyrimethamine, itraconazole, azithromycin",
        "generated_explanation": "A 45-year-old man comes to the physician for a routine health maintenance examination. "
                                 "He was diagnosed with HIV 15 years ago. He was taking triple antiretroviral therapy but "
                                 "stopped a few months ago because he was feeling well. He lives in Wyoming. Vital signs "
                                 "are within normal limits. Cardiopulmonary examination shows no abnormalities. His CD4+ "
                                 "T-lymphocyte count is 47/mm3 (N \u2265 500). The patient currently refuses to restart "
                                 "antiretroviral therapy.\nBased on the patient's condition, the most likely diagnosis or "
                                 "action for the question: 'Which of the following medication regimens is most "
                                 "appropriate at this time?' is: Trimethoprim, sulfamethoxazole, azithromycin."
    },
    {
        "question": "A 38-year-old woman presents to her physician's clinic for recurrent episodes of chest pain that "
                    "wakes her from her sleep. While usually occurring late at night, she has also had similar pains "
                    "during the day at random times, most recently while sitting at her desk in her office and at "
                    "other times while doing the dishes at home. The pain lasts 10\u201315 minutes and resolves "
                    "spontaneously. She is unable to identify any common preceding event to pain onset. The remainder "
                    "of her history is unremarkable and she takes no regular medications. She works as an accountant. "
                    "There is no history of smoking or drug use, however, she does consume 5 alcoholic drinks per "
                    "week. Examination reveals: pulse 70/min, respirations 16/min, and blood pressure 120/70 mm Hg. A "
                    "physical examination is unremarkable. Which of the following would be effective in reducing her "
                    "symptoms?",
        "answer": "Isosorbide dinitrate",
        "answer_idx": "opb",
        "opa": "Aspirin",
        "opb": "Isosorbide dinitrate",
        "opc": "Heparin",
        "opd": "Propranolol",
        "generated_explanation": "A 38-year-old woman presents to her physician's clinic for recurrent episodes of "
                                 "chest pain that wakes her from her sleep. While usually occurring late at night, "
                                 "she has also had similar pains during the day at random times, most recently while "
                                 "sitting at her desk in her office and at other times while doing the dishes at "
                                 "home. The pain lasts 10\u201315 minutes and resolves spontaneously. She is unable "
                                 "to identify any common preceding event to pain onset. The remainder of her history "
                                 "is unremarkable and she takes no regular medications. She works as an accountant. "
                                 "There is no history of smoking or drug use, however, she does consume 5 alcoholic "
                                 "drinks per week. Examination reveals: pulse 70/min, respirations 16/min, "
                                 "and blood pressure 120/70 mm Hg. A physical examination is unremarkable.\nBased on "
                                 "the patient's condition, the most likely diagnosis or action for the question: "
                                 "'Which of the following would be effective in reducing her symptoms?' is: "
                                 "Isosorbide dinitrate."
    }
]
bad_example = {"generated_explanation": "A 28-year-old female comes to the physician's office with a complaint of "
                                        "episodic chest pain. She describes the pain as squeezing and tightness in "
                                        "her chest. This pain has been happening every few days for 3 months. She "
                                        "says there is no association of the pain with food or exercise. She is able "
                                        "to climb up to her fourth floor apartment daily without issue. Her only past "
                                        "medical history is migraines for which she takes appropriate medication. "
                                        "Here temperature is 98.6\u00b0F (37\u00b0C), blood pressure is 120/68 mmHg, "
                                        "pulse is 60/min, respirations are 16/min, and oxygen saturation is 98% on "
                                        "room air. She has no known family history. The patient is not in pain on "
                                        "presentation and EKG in the office is normal. 24-hour ECG monitoring shows "
                                        "transient ST elevations during the episodes of pain that resolve "
                                        "completely.\nBased on the patient's condition, the most likely diagnosis or "
                                        "action for the question: 'The mechanism of this patient's chest pain is most "
                                        "similar to the mechanism behind which of the following?' is: Raynaud's "
                                        "phenomenon."}