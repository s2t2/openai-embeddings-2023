


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
#ORANGES = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']


OPINION_COLORS_MAP = {"Anti-Trump": BLUES[5], "Pro-Trump": REDS[5]}
BOT_COLORS_MAP = {"Human": GREYS[3], "Bot": PURPLES[6]}
Q_COLORS_MAP = {"Normal":GREYS[3], "Q-anon": REDS[6]}

#GROUP_COLORS_MAP = {
#    "Anti-Trump Normal Human": BLUES[3],
#    "Anti-Trump Normal Bot": BLUES[6],
#
#    "Pro-Trump Normal Human": REDS[2],
#    "Pro-Trump Normal Bot": REDS[3],
#
#    "Pro-Trump Q-anon Human": REDS[6],
#    "Pro-Trump Q-anon Bot": REDS[7],
#}
#df["group_color"] = df["group_label"].map(GROUP_COLORS_MAP)

GROUP_COLORS_MAP = {
    "Anti-Trump Human": BLUES[3],
    "Anti-Trump Bot": BLUES[6],

    "Pro-Trump Human": REDS[2],
    "Pro-Trump Bot": REDS[3],

    "Q-anon Human": REDS[6],
    "Q-anon Bot": REDS[7],
}
