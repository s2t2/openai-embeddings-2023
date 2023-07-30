



##from sklearn.metrics import confusion_matrix
##import seaborn as sns
##import matplotlib.pyplot as plt
#import plotly.express as px
#
#
#from app.classification import CLASSES_MAP
#
#
#
#def plot_confusion_matrix(cm, clf, y_col, img_filepath=None):
#    """Params
#
#        cm : an sklearn confusion matrix result
#            ... Confusion matrix whose i-th row and j-th column entry
#            ... indicates the number of samples with true label being i-th class and predicted label being j-th class.
#            ... Interpretation: actual value on rows, predicted value on cols
#
#
#        clf : an sklearn classifier (after it has been trained)
#
#        y_col : the column name of y values (for plot labeling purposes)
#
#        image_filepath : ends with ".png"
#    """
#
#    classes = clf.classes_
#    if y_col in CLASSES_MAP.keys():
#        classes_map = CLASSES_MAP[y_col]
#        class_names = [classes_map[val] for val in classes]
#    else:
#        class_names = classes
#
#
#    title = f"Confusion Matrix ({clf.__class__.__name__})"
#    title += f"<br><sup>Y: '{y_col}'</sup>"
#
#    labels = {"x": "Predicted", "y": "Actual"}
#    fig = px.imshow(cm, x=class_names, y=class_names, height=450, color_continuous_scale="Blues", labels=labels, text_auto=True)
#    fig.update_layout(title={'text': title, 'x':0.485, 'xanchor': 'center'})
#    fig.show()
#
#    if img_filepath:
#        fig.write_image(img_filepath)
#        fig.write_html(img_filepath.replace(".png", ".html"))
#
