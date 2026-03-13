"""Evaluation test queries for retrieval quality measurement.

50 topic-based queries covering major UK parliamentary debate subjects.
Frozen date range: Q1 2025 (2025-01-01 to 2025-03-31).
"""

from datetime import date

DATE_FROM = date(2025, 1, 1)
DATE_TO = date(2025, 3, 31)

EVAL_QUERIES = [
    {"id": "q01_immigration", "text": "immigration"},
    {"id": "q02_nuclear_energy", "text": "nuclear energy"},
    {"id": "q03_housing", "text": "housing"},
    {"id": "q04_hospital_waiting_times", "text": "hospital waiting times"},
    {"id": "q05_net_zero", "text": "net zero"},
    {"id": "q06_gaza", "text": "gaza"},
    {"id": "q07_ukraine_war", "text": "ukraine war"},
    {"id": "q08_tax_rises", "text": "tax rises"},
    {"id": "q09_education", "text": "education"},
    {"id": "q10_policing", "text": "policing"},
    {"id": "q11_defence_spending", "text": "defence spending"},
    {"id": "q12_cost_of_living", "text": "cost of living"},
    {"id": "q13_ai_regulation", "text": "AI regulation"},
    {"id": "q14_social_care", "text": "social care"},
    {"id": "q15_mental_health", "text": "mental health"},
    {"id": "q16_farming_subsidies", "text": "farming subsidies"},
    {"id": "q17_trade_deals", "text": "trade deals"},
    {"id": "q18_energy_prices", "text": "energy prices"},
    {"id": "q19_pensions", "text": "pensions"},
    {"id": "q20_childcare", "text": "childcare"},
    {"id": "q21_broadband", "text": "broadband"},
    {"id": "q22_rail_services", "text": "rail services"},
    {"id": "q23_devolution", "text": "devolution"},
    {"id": "q24_nhs_funding", "text": "NHS funding"},
    {"id": "q25_water_pollution", "text": "water pollution"},
    {"id": "q26_knife_crime", "text": "knife crime"},
    {"id": "q27_free_school_meals", "text": "free school meals"},
    {"id": "q28_universal_credit", "text": "universal credit"},
    {"id": "q29_hs2", "text": "HS2"},
    {"id": "q30_electric_vehicles", "text": "electric vehicles"},
    {"id": "q31_prison_overcrowding", "text": "prison overcrowding"},
    {"id": "q32_channel_crossings", "text": "Channel crossings"},
    {"id": "q33_food_banks", "text": "food banks"},
    {"id": "q34_zero_hours_contracts", "text": "zero hours contracts"},
    {"id": "q35_levelling_up", "text": "levelling up"},
    {"id": "q36_freeports", "text": "freeports"},
    {"id": "q37_local_government_funding", "text": "local government funding"},
    {"id": "q38_council_tax", "text": "council tax"},
    {"id": "q39_send_provision", "text": "SEND provision"},
    {"id": "q40_teacher_recruitment", "text": "teacher recruitment"},
    {"id": "q41_planning_reform", "text": "planning reform"},
    {"id": "q42_green_belt", "text": "green belt"},
    {"id": "q43_national_insurance", "text": "national insurance"},
    {"id": "q44_foreign_aid", "text": "foreign aid"},
    {"id": "q45_rwanda_policy", "text": "Rwanda policy"},
    {"id": "q46_small_boats", "text": "small boats"},
    {"id": "q47_minimum_wage", "text": "minimum wage"},
    {"id": "q48_apprenticeships", "text": "apprenticeships"},
    {"id": "q49_rent_reform", "text": "rent reform"},
    {"id": "q50_northern_powerhouse", "text": "northern powerhouse"},
]
