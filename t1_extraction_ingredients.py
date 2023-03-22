# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"

# Mettre dans cette partie la (les) expression(s) régulière(s)
# que vous utilisez pour analyser les ingrédients
#
# Vos regex ici...
#
regex_1 = re.compile(r"(?P<QUANTITY>[\d,?]+.\w+(?=\s))\s(?P<INGREDIENT>\w.+)")
regex_2 = re.compile(r"(?P<QUANTITY>[\d,?]+\s[a-zÀ-ü|\w.]+\s(à soupe|à café|à thé|à\.s|à\.c|à \.s|à s\.|à c\.)?)"
                     r"\s(?P<INGREDIENT>\w.+)")
regex_3 = re.compile(r"(?P<QUANTITY>[\d,?]*(?:/[\d])*(?: gallon| cuillère à café| cuillère à soupe| cuillère à thé| "
                     r"c\. à thé| tasse)?(\s\w{0,2})?)\s(?P<INGREDIENT>\w.+)")
regex_4 = re.compile(r"(?P<QUANTITY>\d.+?(?=\sd[e']|\))\))\s(?P<INGREDIENT>\w.+)")


def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients


def arrange_ingredient_name(_ingredient):
    ing_clean = " ".join(_ingredient.split())
    ing_clean_1 = ing_clean[ing_clean.startswith('de') and len('de'):]
    ing_clean_2 = ing_clean_1[ing_clean_1.startswith("d’") and len("d’"):]
    return ing_clean_2


def get_ingredients(text):
    # Insérez ici votre code pour l'extraction d'ingrédients.
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    # Vous pouvez ajouter autant de fonctions que vous le souhaitez.
    #
    # IMPORTANT : Ne pas modifier la signature de cette fonction
    #             afin de faciliter notre travail de correction.
    #
    # Votre code ici...
    #
    mass = str
    ingredients = str

    if text.find("(") != -1:
        if re.match(regex_4, text):
            result = re.search(regex_4, text)
            mass = result.group("QUANTITY")
            ingredients = arrange_ingredient_name(result.group("INGREDIENT"))

    elif text.find("tasse") != -1 or text.find("tasses") != -1:
        if re.match(regex_1, text):
            result = re.search(regex_1, text)
            mass = result.group("QUANTITY")
            ingredients = arrange_ingredient_name(result.group("INGREDIENT"))

    else:
        if re.match(regex_2, text):
            result = re.search(regex_2, text)
            mass = result.group("QUANTITY")
            ingredients = arrange_ingredient_name(result.group("INGREDIENT"))

        elif re.match(regex_3, text):
            result = re.search(regex_3, text)
            mass = result.group("QUANTITY")
            ingredients = arrange_ingredient_name(result.group("INGREDIENT"))

        else:
            mass = 0
            ingredients = text

    return mass, ingredients  # À modifier - retourner la paire extraite


if __name__ == '__main__':
    # Vous pouvez modifier cette section
    all_items = load_ingredients(ingredients_fn)
    print(f"la longueur est de {len(all_items)}")
    for item in all_items[:5]:
        print("\t", item)
    print("\nExemples d'extraction")
    for item in all_items:
        quantity, ingredient = get_ingredients(item)
        print("\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(item, quantity, ingredient))
