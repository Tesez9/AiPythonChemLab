import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "khanfs/ChemSolubilityBERTa"
PROPERTY_NAME = "logS"

COMPOUNDS = [
    ("Етанол", "C2H6O", "CCO"),
    ("Оцтова кислота", "C2H4O2", "CC(=O)O"),
    ("Бензен", "C6H6", "c1ccccc1"),
    ("Метанол", "CH4O", "CO"),
    ("Метан", "CH4", "C"),
    ("Етан", "C2H6", "CC"),
    ("Пропан", "C3H8", "CCC"),
    ("Ацетон", "C3H6O", "CC(=O)C"),
    ("Бутанол", "C4H10O", "CCCO"),
    ("Толуен", "C7H8", "Cc1ccccc1")
]

def check_smiles(smiles):
    if smiles == "":
        return False
    allowed = "CNOPSFClBrIcnops0123456789()[]=#+-.@/%\\"
    for ch in smiles:
        if ch not in allowed:
            return False
    count = 0
    for ch in smiles:
        if ch == "(":
            count += 1
        if ch == ")":
            count -= 1
        if count < 0:
            return False
    if count != 0:
        return False
    count = 0
    for ch in smiles:
        if ch == "[":
            count += 1
        if ch == "]":
            count -= 1
        if count < 0:
            return False
    if count != 0:
        return False
    return True

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print("Не вийшло завантажити модель")
        print(e)
        return None, None

def predict(tokenizer, model, smiles):
    inputs = tokenizer(smiles, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    return round(out.logits.item(), 3)

def level(value):
    if value >= -1:
        return "висока розчинність"
    if value >= -3:
        return "помірна розчинність"
    if value >= -5:
        return "низька розчинність"
    return "практично нерозчинна"

def show_result(name, formula, smiles, value):
    print("Речовина:", name)
    print("Формула:", formula)
    print("SMILES:", smiles)
    print("Властивість:", PROPERTY_NAME + " (розчинність у воді, log mol/L)")
    print("Прогноз:", value)
    print("Оцінка:", level(value))

def analyze_one(tokenizer, model):
    print("\n===== ONE MOLECULE =====")
    name = input("Назва речовини: ").strip()
    formula = input("Формула: ").strip()
    smiles = input("SMILES: ").strip()
    if name == "" or smiles == "":
        print("Порожня назва або SMILES")
        return
    if check_smiles(smiles) == False:
        print("Некоректний SMILES")
        return
    try:
        value = predict(tokenizer, model, smiles)
    except Exception as e:
        print("Помилка прогнозування")
        print(e)
        return
    print("")
    show_result(name, formula, smiles, value)

def analyze_all(tokenizer, model):
    print("\n===== ALL MOLECULES =====")
    rows = []
    for name, formula, smiles in COMPOUNDS:
        if check_smiles(smiles) == False:
            print("Пропущено (поганий SMILES):", name)
            continue
        try:
            value = predict(tokenizer, model, smiles)
        except Exception as e:
            print("Помилка на речовині:", name)
            print(e)
            continue
        show_result(name, formula, smiles, value)
        print("----------------------------")
        rows.append((name, formula, smiles, PROPERTY_NAME, value))
    with open("chemistry_results.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["molecule_name", "formula", "smiles", "predicted_property", "predicted_value"])
        for row in rows:
            writer.writerow(row)
    print("Всього проаналізовано:", len(rows))
    print("Результати збережено у chemistry_results.csv")

def main():
    print("Завантаження моделі " + MODEL_NAME)
    tokenizer, model = load_model()
    if tokenizer is None:
        return
    print("Модель готова")
    while True:
        print("""
================================
        AI CHEMISTRY LAB
================================

1. Аналіз однієї речовини
2. Аналіз усіх речовин
0. Вихід
""")
        choice = input("Оберіть пункт: ").strip()
        if choice == "1":
            analyze_one(tokenizer, model)
        elif choice == "2":
            analyze_all(tokenizer, model)
        elif choice == "0":
            print("Програму завершено.")
            break
        else:
            print("Невірний пункт меню.")

if __name__ == "__main__":
    main()
