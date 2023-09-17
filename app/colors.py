


#GREY = "#ccc"
#PURPLE = "#7E57C2"

# colorbrewer scales
# light --> dark
BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
REDS = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
PURPLES = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
GREYS = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000']
GREENS = ["#edf8e9","#c7e9c0","#a1d99b","#74c476","#41ab5d","#238b45","#005a32"]
ORANGES = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
BROWNS = ["#C46200", "#964B00"]
RD_PU = ["#feebe2","#fcc5c0","#fa9fb5","#f768a1","#dd3497","#ae017e","#7a0177"]
PU_RD = ["#f1eef6","#d4b9da","#c994c7","#df65b0","#e7298a","#ce1256","#91003f"]

OPINION_COLORS_MAP = {"Anti-Trump": BLUES[5], "Pro-Trump": REDS[5]}
BOT_COLORS_MAP = {"Human": GREYS[3], "Bot": PURPLES[6]}
Q_COLORS_MAP = {"Normal": GREYS[3], "Q-anon": REDS[6]}
TOXIC_COLORS_MAP = {"Toxic": BROWNS[1], "Normal": GREYS[3]}
FACT_COLORS_MAP = {"High Quality": GREYS[3], "Low Quality": RD_PU[4]}

FOURWAY_COLORS_MAP = {
    "Anti-Trump Human": BLUES[3],
    "Anti-Trump Bot": BLUES[6],

    "Pro-Trump Human": REDS[3],
    "Pro-Trump Bot": REDS[6],
}

COLORS_MAP = {
    "bot_label": BOT_COLORS_MAP,
    "opinion_label": OPINION_COLORS_MAP,
    "q_label": Q_COLORS_MAP,
    "toxic_label": TOXIC_COLORS_MAP,
    "factual_label": FACT_COLORS_MAP,

    "fourway_label": FOURWAY_COLORS_MAP,

    "bom_overall_label": BOT_COLORS_MAP,
    "bom_astroturf_label": BOT_COLORS_MAP,
}


BOT_LABEL_ORDER = ["Human", "Bot"]
CATEGORY_ORDERS = {
    "bot_label": BOT_LABEL_ORDER,
    "bom_overall_label": BOT_LABEL_ORDER,
    "bom_astroturf_label": BOT_LABEL_ORDER,
    "opinion_label": ["Anti-Trump", "Pro-Trump"],
    "q_label": ["Normal", "Q-anon"],

    "toxic_label": ["Normal", "Toxic"],
    "factual_label": ["High Quality", "Low Quality"],

    "fourway_label": list(FOURWAY_COLORS_MAP.keys()),
}
