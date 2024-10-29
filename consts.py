american_to_arabic = {
    'Al': 'ال',
    'Alef': 'ا',
    'Alef': 'أ',
    'Beh': 'ب',
    'Teh': 'ت',
    'Theh': 'ث',
    'Jeem': 'ج',
    'Hah': 'ح',
    'Khah': 'خ',
    'Dal': 'د',
    'Thal': 'ذ',
    'Reh': 'ر',
    'Zain': 'ز',
    'Seen': 'س',
    'Sheen': 'ش',
    'Sad': 'ص',
    'Dad': 'ض',
    'Tah': 'ط',
    'Zah': 'ظ',
    'Ain': 'ع',
    'Ghain': 'غ',
    'Feh': 'ف',
    'Qaf': 'ق',
    'Kaf': 'ك',
    'Lam': 'ل',
    'Meem': 'م',
    'Noon': 'ن',
    'Heh': 'هـ',
    'Waw': 'و',
    'Yeh': 'ي',
    'Teh_Marbuta': 'ة',
    'Laa': 'لا'
}

am_ar_map = {
    'a': 'ا',
    'a': 'أ',
    'b': 'ب',
    'c': 'ج',
    'd': 'د',
    'e': 'ه',
    'f': 'ف',
    'g': 'ج',
    'h': 'ح',
    'i': 'ي',
    'j': 'ج',
    'k': 'ك',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'o': 'و',
    'p': 'ب',
    'q': 'ق',
    'r': 'ر',
    's': 'س',
    't': 'ت',
    'u': 'و',
    'v': 'ف',
    'w': 'و',
    'x': 'إكس',
    'y': 'ي',
    'z': 'ز'
}


ar_am_map = {
    'ا':'a',
    'أ': 'a',
    'ب':'b',
    'ج':'c',
    'د':'d',
    'ه':'e',
    'ف':'f',
    'ج':'g',
    'ح':'h',
    'ي':'i',
    'ج':'j',
    'ك':'k',
    'ل':'l',
    'م':'m',
    'ن':'n',
    'و':'o',
    'ب':'p',
    'ق':'q',
    'ر':'r',
    'س':'s',
    'ت':'t',
    'و':'u',
    'ف':'v',
    'و':'w',
    'إكس':'x',
    'ي':'y',
    'ز':'z',
    }
phonetic_mapping = {
    "American": {
        "A": {"Arabic": "أ", "French": "A", "Chinese": "ā / à", "Brazilian": "A", "Indonesian": "A"},
        "B": {"Arabic": "ب", "French": "B", "Chinese": "b", "Brazilian": "B", "Indonesian": "B"},
        "C": {"Arabic": "ك/ س", "French": "C", "Chinese": "s (si) / k", "Brazilian": "C", "Indonesian": "C"},
        "D": {"Arabic": "د", "French": "D", "Chinese": "d", "Brazilian": "D", "Indonesian": "D"},
        "E": {"Arabic": "إ", "French": "É", "Chinese": "ē / è", "Brazilian": "E", "Indonesian": "E"},
        "F": {"Arabic": "ف", "French": "F", "Chinese": "f", "Brazilian": "F", "Indonesian": "F"},
        "G": {"Arabic": "ج/ غ ", "French": "G", "Chinese": "g / zh", "Brazilian": "G", "Indonesian": "G"},
        "H": {"Arabic": "ه", "French": "H", "Chinese": "h", "Brazilian": "H", "Indonesian": "H"},
        "I": {"Arabic": "إ", "French": "I", "Chinese": "ī / ì", "Brazilian": "I", "Indonesian": "I"},
        "J": {"Arabic": "ج", "French": "J", "Chinese": "j", "Brazilian": "J", "Indonesian": "J"},
        "K": {"Arabic": "ك", "French": "K", "Chinese": "k", "Brazilian": "K", "Indonesian": "K"},
        "L": {"Arabic": "ل", "French": "L", "Chinese": "l", "Brazilian": "L", "Indonesian": "L"},
        "M": {"Arabic": "م", "French": "M", "Chinese": "m", "Brazilian": "M", "Indonesian": "M"},
        "N": {"Arabic": "ن", "French": "N", "Chinese": "n", "Brazilian": "N", "Indonesian": "N"},
        "O": {"Arabic": "و / أُ", "French": "O", "Chinese": "ō / ò", "Brazilian": "O", "Indonesian": "O"},
        "P": {"Arabic": "پ / ب", "French": "P", "Chinese": "p", "Brazilian": "P", "Indonesian": "P"},
        "Q": {"Arabic": "ق", "French": "Q", "Chinese": "q", "Brazilian": "Q", "Indonesian": "Q"},
        "R": {"Arabic": "ر", "French": "R", "Chinese": "r", "Brazilian": "R", "Indonesian": "R"},
        "S": {"Arabic": "س", "French": "S", "Chinese": "s", "Brazilian": "S", "Indonesian": "S"},
        "T": {"Arabic": "ت", "French": "T", "Chinese": "t", "Brazilian": "T", "Indonesian": "T"},
        "U": {"Arabic": "و", "French": "U", "Chinese": "ū / ù", "Brazilian": "U", "Indonesian": "U"},
        "V": {"Arabic": "ڤ / ف", "French": "V", "Chinese": "w", "Brazilian": "V", "Indonesian": "V"},
        "W": {"Arabic": "و", "French": "W", "Chinese": "w", "Brazilian": "W", "Indonesian": "W"},
        "X": {"Arabic": "كس", "French": "X", "Chinese": "x (sh)", "Brazilian": "X (sh)", "Indonesian": "X (s / ks)"},
        "Y": {"Arabic": "ي", "French": "Y", "Chinese": "y", "Brazilian": "Y", "Indonesian": "Y"},
        "Z": {"Arabic": "ز", "French": "Z", "Chinese": "z", "Brazilian": "Z", "Indonesian": "Z"},
    },
    "Arabic": {
        "أ": {"American": "A", "French": "A", "Chinese": "ā / à", "Brazilian": "A", "Indonesian": "A"},
        "ب": {"American": "B", "French": "B", "Chinese": "b", "Brazilian": "B", "Indonesian": "B"},
        "ك": {"American": "K", "French": "K", "Chinese": "k", "Brazilian": "K", "Indonesian": "K"},
        "د": {"American": "D", "French": "D", "Chinese": "d", "Brazilian": "D", "Indonesian": "D"},
        "إ": {"American": "E", "French": "É", "Chinese": "ē / è", "Brazilian": "E", "Indonesian": "E"},
        "ف": {"American": "F", "French": "F", "Chinese": "f", "Brazilian": "F", "Indonesian": "F"},
        "ج": {"American": "G", "French": "G", "Chinese": "j", "Brazilian": "G", "Indonesian": "G"},
        "ه": {"American": "H", "French": "H", "Chinese": "h", "Brazilian": "H", "Indonesian": "H"},
        "و": {"American": "U", "French": "O", "Chinese": "w", "Brazilian": "U", "Indonesian": "U"},
        "ي": {"American": "Y", "French": "Y", "Chinese": "y", "Brazilian": "Y", "Indonesian": "Y"},
        "ر": {"American": "R", "French": "R", "Chinese": "r", "Brazilian": "R", "Indonesian": "R"},
        "س": {"American": "S", "French": "S", "Chinese": "s", "Brazilian": "S", "Indonesian": "S"},
        "ت": {"American": "T", "French": "T", "Chinese": "t", "Brazilian": "T", "Indonesian": "T"},
        "ز": {"American": "Z", "French": "Z", "Chinese": "z", "Brazilian": "Z", "Indonesian": "Z"},
    },
    "French":{
    "A": {"Arabic": "أ", "American": "A", "Chinese": "ā / à", "Brazilian": "A", "Indonesian": "A"},
    "B": {"Arabic": "ب", "American": "B", "Chinese": "b", "Brazilian": "B", "Indonesian": "B"},
    "C": {"Arabic": "ك (k) or س (s)", "American": "C", "Chinese": "s (si) / k", "Brazilian": "C", "Indonesian": "C"},
    "D": {"Arabic": "د", "American": "D", "Chinese": "d", "Brazilian": "D", "Indonesian": "D"},
    "E": {"Arabic": "إ", "American": "E", "Chinese": "ē / è", "Brazilian": "E", "Indonesian": "E"},
    "F": {"Arabic": "ف", "American": "F", "Chinese": "f", "Brazilian": "F", "Indonesian": "F"},
    "G": {"Arabic": "ج (j) or غ (g)", "American": "G", "Chinese": "g / zh", "Brazilian": "G", "Indonesian": "G"},
    "H": {"Arabic": "ه", "American": "H", "Chinese": "h", "Brazilian": "H", "Indonesian": "H"},
    "I": {"Arabic": "إ", "American": "I", "Chinese": "ī / ì", "Brazilian": "I", "Indonesian": "I"},
    "J": {"Arabic": "ج", "American": "J", "Chinese": "j", "Brazilian": "J", "Indonesian": "J"},
    "K": {"Arabic": "ك", "American": "K", "Chinese": "k", "Brazilian": "K", "Indonesian": "K"},
    "L": {"Arabic": "ل", "American": "L", "Chinese": "l", "Brazilian": "L", "Indonesian": "L"},
    "M": {"Arabic": "م", "American": "M", "Chinese": "m", "Brazilian": "M", "Indonesian": "M"},
    "N": {"Arabic": "ن", "American": "N", "Chinese": "n", "Brazilian": "N", "Indonesian": "N"},
    "O": {"Arabic": "و / أُ", "American": "O", "Chinese": "ō / ò", "Brazilian": "O", "Indonesian": "O"},
    "P": {"Arabic": "پ or ب", "American": "P", "Chinese": "p", "Brazilian": "P", "Indonesian": "P"},
    "Q": {"Arabic": "ق", "American": "Q", "Chinese": "q", "Brazilian": "Q", "Indonesian": "Q"},
    "R": {"Arabic": "ر", "American": "R", "Chinese": "r", "Brazilian": "R", "Indonesian": "R"},
    "S": {"Arabic": "س", "American": "S", "Chinese": "s", "Brazilian": "S", "Indonesian": "S"},
    "T": {"Arabic": "ت", "American": "T", "Chinese": "t", "Brazilian": "T", "Indonesian": "T"},
    "U": {"Arabic": "و", "American": "U", "Chinese": "ū / ù", "Brazilian": "U", "Indonesian": "U"},
    "V": {"Arabic": "ڤ or ف", "American": "V", "Chinese": "w", "Brazilian": "V", "Indonesian": "V"},
    "W": {"Arabic": "و", "American": "W", "Chinese": "w", "Brazilian": "W", "Indonesian": "W"},
    "X": {"Arabic": "كس", "American": "X", "Chinese": "x (sh)", "Brazilian": "X (sh)", "Indonesian": "X (s / ks)"},
    "Y": {"Arabic": "ي", "American": "Y", "Chinese": "y", "Brazilian": "Y", "Indonesian": "Y"},
    "Z": {"Arabic": "ز", "American": "Z", "Chinese": "z", "Brazilian": "Z", "Indonesian": "Z"},
    "É": {"Arabic": "إ", "American": "E", "Chinese": "ē / è", "Brazilian": "E", "Indonesian": "E"},
},
"Chinese":{
    "ā": {"Arabic": "أ", "American": "A", "French": "A", "Brazilian": "A", "Indonesian": "A"},
    "b": {"Arabic": "ب", "American": "B", "French": "B", "Brazilian": "B", "Indonesian": "B"},
    "c": {"Arabic": "س (s)", "American": "C", "French": "C", "Brazilian": "C", "Indonesian": "C"},
    "d": {"Arabic": "د", "American": "D", "French": "D", "Brazilian": "D", "Indonesian": "D"},
    "e": {"Arabic": "إ", "American": "E", "French": "E", "Brazilian": "E", "Indonesian": "E"},
    "f": {"Arabic": "ف", "American": "F", "French": "F", "Brazilian": "F", "Indonesian": "F"},
    "g": {"Arabic": "غ", "American": "G", "French": "G", "Brazilian": "G", "Indonesian": "G"},
    "h": {"Arabic": "ه", "American": "H", "French": "H", "Brazilian": "H", "Indonesian": "H"},
    "i": {"Arabic": "إ", "American": "I", "French": "I", "Brazilian": "I", "Indonesian": "I"},
    "j": {"Arabic": "ج", "American": "J", "French": "J", "Brazilian": "J", "Indonesian": "J"},
    "k": {"Arabic": "ك", "American": "K", "French": "K", "Brazilian": "K", "Indonesian": "K"},
    "l": {"Arabic": "ل", "American": "L", "French": "L", "Brazilian": "L", "Indonesian": "L"},
    "m": {"Arabic": "م", "American": "M", "French": "M", "Brazilian": "M", "Indonesian": "M"},
    "n": {"Arabic": "ن", "American": "N", "French": "N", "Brazilian": "N", "Indonesian": "N"},
    "o": {"Arabic": "أُ", "American": "O", "French": "O", "Brazilian": "O", "Indonesian": "O"},
    "p": {"Arabic": "ب", "American": "P", "French": "P", "Brazilian": "P", "Indonesian": "P"},
    "q": {"Arabic": "ق", "American": "Q", "French": "Q", "Brazilian": "Q", "Indonesian": "Q"},
    "r": {"Arabic": "ر", "American": "R", "French": "R", "Brazilian": "R", "Indonesian": "R"},
    "s": {"Arabic": "س", "American": "S", "French": "S", "Brazilian": "S", "Indonesian": "S"},
    "t": {"Arabic": "ت", "American": "T", "French": "T", "Brazilian": "T", "Indonesian": "T"},
    "u": {"Arabic": "و", "American": "U", "French": "U", "Brazilian": "U", "Indonesian": "U"},
    "v": {"Arabic": "ڤ", "American": "V", "French": "V", "Brazilian": "V", "Indonesian": "V"},
    "w": {"Arabic": "و", "American": "W", "French": "W", "Brazilian": "W", "Indonesian": "W"},
    "x": {"Arabic": "ك (k)", "American": "X", "French": "X", "Brazilian": "X", "Indonesian": "X"},
    "y": {"Arabic": "ي", "American": "Y", "French": "Y", "Brazilian": "Y", "Indonesian": "Y"},
    "z": {"Arabic": "ز", "American": "Z", "French": "Z", "Brazilian": "Z", "Indonesian": "Z"},
},
"Brazilian":{
    "A": {"Arabic": "أ", "American": "A", "French": "A", "Chinese": "ā", "Indonesian": "A"},
    "B": {"Arabic": "ب", "American": "B", "French": "B", "Chinese": "b", "Indonesian": "B"},
    "C": {"Arabic": "س (s) or ك (k)", "American": "C", "French": "C", "Chinese": "c", "Indonesian": "C"},
    "D": {"Arabic": "د", "American": "D", "French": "D", "Chinese": "d", "Indonesian": "D"},
    "E": {"Arabic": "إ", "American": "E", "French": "E", "Chinese": "ē", "Indonesian": "E"},
    "F": {"Arabic": "ف", "American": "F", "French": "F", "Chinese": "f", "Indonesian": "F"},
    "G": {"Arabic": "غ or ج (g)", "American": "G", "French": "G", "Chinese": "g", "Indonesian": "G"},
    "H": {"Arabic": "ه", "American": "H", "French": "H", "Chinese": "h", "Indonesian": "H"},
    "I": {"Arabic": "إ", "American": "I", "French": "I", "Chinese": "ī", "Indonesian": "I"},
    "J": {"Arabic": "ج", "American": "J", "French": "J", "Chinese": "j", "Indonesian": "J"},
    "K": {"Arabic": "ك", "American": "K", "French": "K", "Chinese": "k", "Indonesian": "K"},
    "L": {"Arabic": "ل", "American": "L", "French": "L", "Chinese": "l", "Indonesian": "L"},
    "M": {"Arabic": "م", "American": "M", "French": "M", "Chinese": "m", "Indonesian": "M"},
    "N": {"Arabic": "ن", "American": "N", "French": "N", "Chinese": "n", "Indonesian": "N"},
    "O": {"Arabic": "أُ", "American": "O", "French": "O", "Chinese": "ō", "Indonesian": "O"},
    "P": {"Arabic": "ب", "American": "P", "French": "P", "Chinese": "p", "Indonesian": "P"},
    "Q": {"Arabic": "ق", "American": "Q", "French": "Q", "Chinese": "q", "Indonesian": "Q"},
    "R": {"Arabic": "ر or خ (h)", "American": "R", "French": "R", "Chinese": "r", "Indonesian": "R"},
    "S": {"Arabic": "س", "American": "S", "French": "S", "Chinese": "s", "Indonesian": "S"},
    "T": {"Arabic": "ت", "American": "T", "French": "T", "Chinese": "t", "Indonesian": "T"},
    "U": {"Arabic": "و", "American": "U", "French": "U", "Chinese": "ū", "Indonesian": "U"},
    "V": {"Arabic": "ڤ or ف", "American": "V", "French": "V", "Chinese": "v", "Indonesian": "V"},
    "W": {"Arabic": "و", "American": "W", "French": "W", "Chinese": "w", "Indonesian": "W"},
    "X": {"Arabic": "ك (k) or ش (sh)", "American": "X", "French": "X", "Chinese": "x", "Indonesian": "X"},
    "Y": {"Arabic": "ي", "American": "Y", "French": "Y", "Chinese": "y", "Indonesian": "Y"},
    "Z": {"Arabic": "ز", "American": "Z", "French": "Z", "Chinese": "z", "Indonesian": "Z"},

},
"Indonisian":{
    "A": {"Arabic": "أ", "American": "A", "French": "A", "Chinese": "ā", "Brazilian": "A"},
    "B": {"Arabic": "ب", "American": "B", "French": "B", "Chinese": "b", "Brazilian": "B"},
    "C": {"Arabic": "س (s)", "American": "C", "French": "C", "Chinese": "c", "Brazilian": "C"},
    "D": {"Arabic": "د", "American": "D", "French": "D", "Chinese": "d", "Brazilian": "D"},
    "E": {"Arabic": "إ", "American": "E", "French": "E", "Chinese": "ē", "Brazilian": "E"},
    "F": {"Arabic": "ف", "American": "F", "French": "F", "Chinese": "f", "Brazilian": "F"},
    "G": {"Arabic": "غ or ج (g)", "American": "G", "French": "G", "Chinese": "g", "Brazilian": "G"},
    "H": {"Arabic": "ه", "American": "H", "French": "H", "Chinese": "h", "Brazilian": "H"},
    "I": {"Arabic": "إ", "American": "I", "French": "I", "Chinese": "ī", "Brazilian": "I"},
    "J": {"Arabic": "ج", "American": "J", "French": "J", "Chinese": "j", "Brazilian": "J"},
    "K": {"Arabic": "ك", "American": "K", "French": "K", "Chinese": "k", "Brazilian": "K"},
    "L": {"Arabic": "ل", "American": "L", "French": "L", "Chinese": "l", "Brazilian": "L"},
    "M": {"Arabic": "م", "American": "M", "French": "M", "Chinese": "m", "Brazilian": "M"},
    "N": {"Arabic": "ن", "American": "N", "French": "N", "Chinese": "n", "Brazilian": "N"},
    "O": {"Arabic": "أُ", "American": "O", "French": "O", "Chinese": "ō", "Brazilian": "O"},
    "P": {"Arabic": "ب", "American": "P", "French": "P", "Chinese": "p", "Brazilian": "P"},
    "Q": {"Arabic": "ق", "American": "Q", "French": "Q", "Chinese": "q", "Brazilian": "Q"},
    "R": {"Arabic": "ر", "American": "R", "French": "R", "Chinese": "r", "Brazilian": "R"},
    "S": {"Arabic": "س", "American": "S", "French": "S", "Chinese": "s", "Brazilian": "S"},
    "T": {"Arabic": "ت", "American": "T", "French": "T", "Chinese": "t", "Brazilian": "T"},
    "U": {"Arabic": "و", "American": "U", "French": "U", "Chinese": "ū", "Brazilian": "U"},
    "V": {"Arabic": "ڤ or ف", "American": "V", "French": "V", "Chinese": "v", "Brazilian": "V"},
    "W": {"Arabic": "و", "American": "W", "French": "W", "Chinese": "w", "Brazilian": "W"},
    "X": {"Arabic": "ك (k)", "American": "X", "French": "X", "Chinese": "x", "Brazilian": "X"},
    "Y": {"Arabic": "ي", "American": "Y", "French": "Y", "Chinese": "y", "Brazilian": "Y"},
    "Z": {"Arabic": "ز", "American": "Z", "French": "Z", "Chinese": "z", "Brazilian": "Z"},
}
}