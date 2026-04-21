import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── Synthetic Training Data ─────────────────────────────────────────────────
# Features: calories, protein, carbs, fats, fiber, sugar, sodium, risk_tag_count
# Label: 0=Avoid, 1=Moderate, 2=Best Choice
TRAINING_DATA = [
    # calories, protein, carbs, fats, fiber, sugar, sodium, risk_count → label
    [100, 5,  18, 1,  6,  8,  120, 0, 2],  # Best Choice
    [160, 4,  36, 1,  6,  22, 80,  0, 2],  # Best Choice (smoothie)
    [180, 24, 6,  7,  2,  3,  220, 1, 2],  # Best Choice (egg white)
    [200, 4,  46, 4,  7,  14, 160, 0, 2],  # Best Choice (sweet potato)
    [240, 16, 38, 4,  10, 6,  380, 0, 2],  # Best Choice (lentil soup)
    [280, 8,  52, 6,  8,  14, 80,  1, 1],  # Moderate (oatmeal)
    [280, 10, 48, 6,  7,  4,  240, 1, 1],  # Moderate (roti)
    [290, 8,  32, 16, 8,  3,  310, 0, 2],  # Best Choice (avo toast)
    [300, 12, 54, 4,  6,  16, 240, 2, 1],  # Moderate (cereal)
    [310, 18, 42, 8,  3,  22, 120, 1, 1],  # Moderate (parfait)
    [310, 14, 54, 6,  6,  2,  180, 0, 2],  # Best Choice (khichdi)
    [320, 35, 12, 14, 4,  3,  280, 0, 2],  # Best Choice (chicken salad)
    [340, 28, 38, 8,  6,  18, 180, 2, 1],  # Moderate (protein bowl)
    [380, 18, 52, 12, 10, 6,  320, 1, 2],  # Best Choice (quinoa)
    [380, 42, 2,  22, 0,  0,  280, 1, 2],  # Best Choice (salmon)
    [380, 20, 12, 26, 4,  5,  480, 2, 1],  # Moderate (palak paneer)
    [410, 22, 14, 28, 2,  6,  520, 2, 1],  # Moderate (paneer tikka)
    [420, 18, 72, 8,  9,  4,  340, 1, 1],  # Moderate (dal rice)
    [420, 26, 56, 10, 3,  6,  840, 2, 1],  # Moderate (sushi)
    [480, 20, 82, 10, 12, 5,  420, 1, 1],  # Moderate (rajma)
    [520, 10, 76, 20, 6,  8,  820, 3, 0],  # Avoid (pav bhaji)
    [520, 16, 82, 14, 4,  8,  620, 2, 0],  # Avoid (pasta)
    [560, 34, 18, 38, 2,  8,  780, 2, 0],  # Avoid (butter chicken)
    [580, 32, 68, 18, 2,  4,  620, 2, 1],  # Moderate (biryani)
    [620, 36, 24, 40, 1,  1,  980, 3, 0],  # Avoid (fried chicken)
    [680, 28, 48, 38, 2,  12, 1100,4, 0],  # Avoid (cheeseburger)
    [680, 18, 92, 28, 8,  6,  720, 3, 0],  # Avoid (chole bhature)
    [720, 26, 88, 28, 3,  8,  980, 4, 0],  # Avoid (pizza)
    [260, 4,  34, 14, 2,  1,  420, 3, 0],  # Avoid (samosa)
    [400, 4,  50, 22, 3,  0,  720, 3, 0],  # Avoid (fries)
    [380, 5,  52, 18, 2,  38, 180, 4, 0],  # Avoid (brownie)
    [140, 0,  38, 0,  0,  38, 45,  4, 0],  # Avoid (soda)
    [380, 8,  52, 16, 1,  2,  1580,4, 0],  # Avoid (instant noodles)
    [240, 6,  42, 5,  1,  36, 80,  2, 0],  # Avoid (mango lassi high sugar)
    [320, 0,  56, 8,  2,  42, 120, 2, 0],  # Avoid (milkshake high sugar)
    [160, 12, 24, 2,  8,  4,  180, 0, 2],  # Best Choice (sprouts)
    [200, 5,  28, 10, 5,  20, 60,  1, 1],  # Moderate (apple+almond)
    [260, 24, 22, 8,  3,  4,  680, 1, 1],  # Moderate (chicken soup)
    [280, 18, 16, 14, 4,  4,  620, 1, 1],  # Moderate (tofu)
    [220, 6,  38, 6,  4,  3,  280, 1, 1],  # Moderate (upma)
    [240, 9,  48, 2,  5,  4,  320, 0, 2],  # Best Choice (idli sambar)
]

# ─── ML Model ─────────────────────────────────────────────────────────────────
X = np.array([row[:8] for row in TRAINING_DATA], dtype=float)
y = np.array([row[8] for row in TRAINING_DATA])

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
LABEL_MAP = {2: 'Best Choice', 1: 'Moderate', 0: 'Avoid'}

print("[OK] Random Forest model trained on", len(X), "samples")

# ─── Allergy Keywords ─────────────────────────────────────────────────────────
ALLERGY_KEYWORDS = {
    'nuts': ['nut', 'almond', 'cashew', 'walnut', 'peanut', 'pistachio'],
    'dairy': ['dairy', 'milk', 'cheese', 'butter', 'cream', 'yogurt', 'paneer', 'whey'],
    'gluten': ['gluten', 'wheat', 'bread', 'pasta', 'flour', 'noodle', 'crouton', 'semolina'],
    'eggs': ['egg'],
    'fish': ['fish', 'salmon', 'tuna', 'sushi'],
    'soy': ['soy', 'tofu'],
    'sesame': ['sesame', 'tahini'],
}

CONDITION_RULES = {
    'diabetes': {'high_sugar_threshold': 20, 'fiber_bonus': True},
    'prediabetes': {'high_sugar_threshold': 15, 'fiber_bonus': True},
    'hypertension': {'high_sodium_threshold': 500},
    'high bp': {'high_sodium_threshold': 500},
    'pcos': {'processed_penalty': True, 'fiber_bonus': True},
    'cholesterol': {'high_fat_threshold': 20},
    'obesity': {'high_calorie_threshold': 400},
}

# ─── Core Scorer ──────────────────────────────────────────────────────────────
def score_food(food, profile, context):
    conditions = [c.lower() for c in (profile.get('conditions') or [])]
    allergies = [a.lower() for a in (profile.get('allergies') or [])]
    goal = (profile.get('goal') or 'maintenance').lower()
    meal_type = (context.get('mealType') or 'lunch').lower()
    activity = (context.get('activityLevel') or 'moderate').lower()

    ingredients_str = ' '.join(food.get('ingredients', [])).lower()
    risk_tags_str = ' '.join(food.get('riskTags', [])).lower()

    # ── Allergy hard block ───────────────────────────────────────────────────
    for allergy in allergies:
        keywords = ALLERGY_KEYWORDS.get(allergy, [allergy])
        if any(kw in ingredients_str or kw in risk_tags_str for kw in keywords):
            return {'score': 0, 'label': 'Avoid', 'reason': f'Contains allergen: {allergy}'}

    # ── Build feature vector ─────────────────────────────────────────────────
    risk_count = len(food.get('riskTags', []))
    features = np.array([[
        food.get('calories', 300),
        food.get('protein', 10),
        food.get('carbs', 30),
        food.get('fats', 10),
        food.get('fiber', 3),
        food.get('sugar', 5),
        food.get('sodium', 300),
        risk_count
    ]], dtype=float)

    # ── ML base prediction ───────────────────────────────────────────────────
    proba = rf_model.predict_proba(features)[0]
    class_indices = rf_model.classes_
    base_score = sum(proba[i] * (class_indices[i] / 2.0) for i in range(len(class_indices))) * 100

    score = base_score  # 0-100 range

    # ── Medical condition adjustments ────────────────────────────────────────
    for cond, rules in CONDITION_RULES.items():
        if cond not in conditions:
            continue
        if 'high_sugar_threshold' in rules and food.get('sugar', 0) > rules['high_sugar_threshold']:
            score -= 20
        if 'high_sodium_threshold' in rules and food.get('sodium', 0) > rules['high_sodium_threshold']:
            score -= 18
        if 'high_fat_threshold' in rules and food.get('fats', 0) > rules['high_fat_threshold']:
            score -= 15
        if 'high_calorie_threshold' in rules and food.get('calories', 0) > rules['high_calorie_threshold']:
            score -= 12
        if rules.get('fiber_bonus') and food.get('fiber', 0) > 5:
            score += 10
        if rules.get('processed_penalty') and 'processed' in risk_tags_str:
            score -= 10

    # ── Goal adjustments ────────────────────────────────────────────────────
    if goal == 'weight_loss':
        if food.get('calories', 0) > 500: score -= 18
        elif food.get('calories', 0) < 280: score += 12
        if food.get('fiber', 0) > 6: score += 8
        if 'fried' in risk_tags_str: score -= 14
    elif goal == 'muscle_gain':
        protein = food.get('protein', 0)
        if protein > 30: score += 22
        elif protein > 20: score += 12
        elif protein < 10: score -= 12
    elif goal == 'maintenance':
        if food.get('calories', 0) > 700: score -= 10

    # ── Context adjustments ─────────────────────────────────────────────────
    if meal_type == 'breakfast' and food.get('category') == 'breakfast':
        score += 8
    if meal_type == 'dinner':
        if activity == 'rest' and food.get('calories', 0) > 550:
            score -= 14
        if food.get('category') == 'dinner':
            score += 6

    if activity == 'active':
        if food.get('carbs', 0) > 40: score += 7
        if food.get('protein', 0) > 20: score += 7
    elif activity == 'rest':
        if food.get('calories', 0) > 500: score -= 10

    # ── Risk tag penalties ──────────────────────────────────────────────────
    if 'very high sodium' in risk_tags_str: score -= 18
    if 'processed' in risk_tags_str: score -= 8
    if 'zero nutrition' in risk_tags_str: score -= 20

    # ── Positive nutrition bonuses ──────────────────────────────────────────
    if len(food.get('riskTags', [])) == 0: score += 8
    benefits_str = ' '.join(food.get('benefits', [])).lower()
    if 'antioxidant' in benefits_str: score += 4
    if 'omega' in benefits_str: score += 4

    # ── Clamp ──────────────────────────────────────────────────────────────
    score = max(0, min(100, score))
    score = round(score)

    if score >= 70:
        label = 'Best Choice'
    elif score >= 45:
        label = 'Moderate'
    else:
        label = 'Avoid'

    return {'score': score, 'label': label}


def score_foods(foods, profile, context):
    results = {}
    for food in foods:
        results[food['id']] = score_food(food, profile, context)
    return results
