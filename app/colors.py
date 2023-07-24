


#GREY = "#ccc"
#PURPLE = "#7E57C2"

# light --> dark
#BLUES = ["#3498DB", "#2E86C1", "#2874A6"]
#REDS = ["#D98880", "#E6B0AA", "#C0392B", "#B03A2E", "#922B21"]

# colorbrewer scales
BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
REDS = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
PURPLES = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
GREYS = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000']
#GREENS = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
ORANGES = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']


OPINION_COLORS_MAP = {"Anti-Trump": BLUES[5], "Pro-Trump": REDS[5]}
BOT_COLORS_MAP = {"Human": GREYS[3], "Bot": PURPLES[6]}
Q_COLORS_MAP = {"Normal": GREYS[3], "Q-anon": REDS[6]}

FOURWAY_COLORS_MAP = {
    "Anti-Trump Human": BLUES[3],
    "Anti-Trump Bot": BLUES[6],

    "Pro-Trump Human": REDS[3],
    "Pro-Trump Bot": REDS[6],
}

SIXWAY_COLORS_MAP = {
    "Anti-Trump Human": BLUES[3],
    "Anti-Trump Bot": BLUES[6],

    "Pro-Trump Human": REDS[3],
    "Pro-Trump Bot": REDS[6],

    "Q-anon Human": REDS[4], # "Pro-Trump Q-anon Human"
    "Q-anon Bot": REDS[7], # "Pro-Trump Q-anon Bot"
}


COLORS_MAP = {
    "bot_label": BOT_COLORS_MAP,
    "opinion_label": OPINION_COLORS_MAP,
    "q_label": Q_COLORS_MAP,
    "fourway_label": FOURWAY_COLORS_MAP,
    "sixway_label": SIXWAY_COLORS_MAP,
    "bom_overall_label": BOT_COLORS_MAP,
    "bom_astroturf_label": BOT_COLORS_MAP
}


BOT_LABEL_ORDER = ["Human", "Bot"]
CATEGORY_ORDERS = {
    "bot_label": BOT_LABEL_ORDER,
    "bom_overall_label": BOT_LABEL_ORDER,
    "bom_astroturf_label": BOT_LABEL_ORDER,
    "opinion_label": ["Anti-Trump", "Pro-Trump"],
    "q_label": ["Normal", "Q-anon"],
    "fourway_label": list(FOURWAY_COLORS_MAP.keys()),
    "sixway_label": list(SIXWAY_COLORS_MAP.keys()),
}
