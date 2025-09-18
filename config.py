import re
from nltk.corpus import stopwords
PATTERNS = [
    # Fixing common negation typos
    ("didnt_fix", re.compile(r"\bdidn[’]?t\b"), "didn't"),
    ("cant_fix", re.compile(r"\bcan[’]?t\b"),    "can't"),
    ("hadnt_fix", re.compile(r"\bhadn[’]?t\b"),   "hadn't"),
    ("dont_fix", re.compile(r"\bdon[’]?t\b"),    "don't"),
    ("isnt_fix", re.compile(r"\bisn[’]?t\b"),    "isn't"),
    ("doesnt_fix", re.compile(r"\bdoesn[’]?t\b"),  "doesn't"),
    ("wasnt_fix", re.compile(r"\bwasn[’]?t\b"),   "wasn't"),
    ("werent_fix", re.compile(r"\bweren[’]?t\b"),  "weren't"),
    ("hasnt_fix", re.compile(r"\bhasn[’]?t\b"),   "hasn't"),
    ("havent_fix", re.compile(r"\bhaven[’]?t\b"),  "haven't"),
    ("wont_fix", re.compile(r"\bwon[’]?t\b"),    "won't"),
    ("wouldnt_fix", re.compile(r"\bwouldn[’]?t\b"), "wouldn't"),
    ("couldnt_fix", re.compile(r"\bcouldn[’]?t\b"), "couldn't"),
    ("shouldnt_fix",re.compile(r"\bshouldn[’]?t\b"),"shouldn't"),
    ("aint_fix", re.compile(r"\bain[’]?t\b"),    "ain't"),
    ("needn’t_fix", re.compile(r"\bneedn[’]?t\b"),    "needn't"),
    ("musnt_variant_fix", re.compile(r"\bmusn['’]?t\b"), "mustn't"),

    
    # General cleaning patterns
    ("html_tags", re.compile(r"<[^>]+>"), ""),
    ("ctrl_ws", re.compile(r"[\r\n\t]+"), " "),
    ("urls", re.compile(r"https?://\S+|www\.\S+"), " URL "),
    ("emails", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), " EMAIL "),
    ("handles", re.compile(r"(?<!\w)@[A-Za-z0-9_]+"), " USER "),
    ("numbers", re.compile(r"\b\w*\d+\w*\b"), " NUM "),
    ("quests", re.compile(r"\?{2,}"), "?"),
    ("commas", re.compile(r",\s*,+"), ","),
    ("boundary_dashes", re.compile(r"(?<=\s)[-–—](?=\w)|(?<=\w)[-–—](?=\s)"), " - "),
    ("ellipses", re.compile(r"\.{2,}"), " "),
    ("elong", re.compile(r"([A-Za-z])\1{2,}"), r"\1\1"),
    ("quote_wrapped_word",
     re.compile(
         r'(?<!\w)["“”\'’]\s*('
         r'(?:[A-Za-z]+(?:\'[A-Za-z]+)?(?:-[A-Za-z]+(?:\'[A-Za-z]+)?)*)'
         r'(?:\s+(?:[A-Za-z]+(?:\'[A-Za-z]+)?(?:-[A-Za-z]+(?:\'[A-Za-z]+)?)*))*'
         r')\s*["“”\'’](?!\w)'
     ),
     r"\1"),
    ("html_entities", re.compile(r"&[A-Za-z]+;"), " "),
    ("ampersand_to_and", re.compile(r"&(?![A-Za-z]+;)"), " and "),
    ("period_letter_space", re.compile(r"(?<=\w)\.(?=[A-Za-z])"), ". "),
  # ("split_letter_hyphens", re.compile(r"(?i)(?<=[a-z])[–—-](?=[a-z])")," "),
    ("nonword_symbols", re.compile(r"[^A-Za-z0-9\s.,!?…'\-]"), " "),
    ("multi_space", re.compile(r"\s{2,}"), " ")
]

NEGATORS = {"not","no","never","nor","without"}

ARTIFACTS = {"ll", "ve", "re", "m", "d", "s", "t", "'m", "'ll", "'ve", "'re", "'s", "'t", "'d"}

AUXILIARIES = {
    "do","does","did",
    "have","has","had",
    "am","is","are","was","were","be","been","being",
    "can","could","may","might","must","shall","should","will","would",
    "need","must","wo","sha","ca","ai"               
}

PUNCTUATION = {".", "!", "?", "-", ","}
ABBREVIATIONS = {"p.", "s.", "etc.", "dr.", "mr.", "mrs.", "ms.", "prof.", "vs."}
PLACEHOLDERS = {"NUM", "URL", "EMAIL", "USER"}
COMMON_WORDS = {"us", "one", "get", "make", "give", "put", "see", "know", "find"}
BASE= set(stopwords.words("english"))
